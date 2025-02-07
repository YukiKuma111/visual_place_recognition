import pickle
import cv2
import numpy as np
import rospy
import open3d as o3d
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
import transforms3d as tf


class FeatureMatcher:
    def __init__(
        self,
        feature_db_path,
        global_pc_path,
        image_topic,
        pose_topic,
        cloud_topic,
        opt_pose_topic,
        voxel_size=2.0,  # 体素大小，用于降采样
        visualize=False,
        width_crop=150,
        icp_opt=False
    ):
        self.feature_db_path = feature_db_path
        self.global_pc_path = global_pc_path
        self.image_topic = image_topic
        self.cloud_topic = cloud_topic
        self.opt_pose_topic = opt_pose_topic
        self.voxel_size = voxel_size  # 降采样体素大小
        self.features_database = []
        self.bridge = CvBridge()
        self.visualize = visualize
        self.width_crop = width_crop
        self.icp_opt = icp_opt
        self.orb = cv2.ORB_create(
            nfeatures=1500, scaleFactor=1.1, nlevels=12, edgeThreshold=15
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.pose_publisher = rospy.Publisher(pose_topic, PoseStamped, queue_size=10)
        self.opt_pose_publisher = rospy.Publisher(
            opt_pose_topic, PoseStamped, queue_size=10
        )

        # 读取全局地图点云
        self.global_map = o3d.io.read_point_cloud(self.global_pc_path)
        rospy.loginfo(
            "Loaded global map with {} points".format(len(self.global_map.points))
        )

        self.current_cloud = None  # 存储最新的点云数据
        self.downsampled_cloud = None  # 存储降采样后的点云
        self.load_feature_database()

        # 订阅点云数据
        rospy.Subscriber(cloud_topic, PointCloud2, self.cloud_callback)

        # 记录上次保存 pcd 的时间
        self.last_save_time = time.time()
    
    def image_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像
            new_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            height, width, _ = new_image.shape
            if width > 500:
                cropped_image = new_image[:, self.width_crop : width - self.width_crop]
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # 提取实时图像特征
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            if descriptors is None:
                rospy.logwarn("No features extracted from the live image!")
                return

            # 进行匹配
            best_match = None
            min_distance = float("inf")
            best_matches = []

            for data in self.features_database:
                if data["descriptors"] is None:
                    continue

                matches = self.bf.match(descriptors, data["descriptors"])
                avg_distance = sum([m.distance for m in matches]) / len(matches)

                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_match = data
                    best_matches = matches

            if best_match:
                rospy.loginfo(f"Best matching image timestamp: {best_match['timestamp']}")

                # 计算初始位姿
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.from_sec(best_match["timestamp"])
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = best_match["pose"]["translation"][0]
                pose_msg.pose.position.y = best_match["pose"]["translation"][1]
                pose_msg.pose.position.z = best_match["pose"]["translation"][2]
                pose_msg.pose.orientation.x = best_match["pose"]["rotation"][0]
                pose_msg.pose.orientation.y = best_match["pose"]["rotation"][1]
                pose_msg.pose.orientation.z = best_match["pose"]["rotation"][2]
                pose_msg.pose.orientation.w = best_match["pose"]["rotation"][3]
                self.pose_publisher.publish(pose_msg)

                # 执行 ICP 优化位姿
                if self.current_cloud is not None and self.icp_opt:
                    self.optimize_pose(pose_msg)

                # 可视化（如果启用）
                if self.visualize:
                    curr_image = cv2.drawKeypoints(
                        cropped_image, keypoints, None, color=(0, 255, 0)
                    )
                    cv2.imshow("Keypoints with Matches", curr_image)
                    cv2.waitKey(1)
            else:
                rospy.loginfo("No suitable matches found!")
        except Exception as e:
            rospy.logerr(f"Error while processing image: {e}")


    def load_feature_database(self):
        with open(self.feature_db_path, "rb") as f:
            self.features_database = pickle.load(f)

        for data in self.features_database:
            if data["descriptors"] is not None:
                data["descriptors"] = np.array(data["descriptors"], dtype=np.uint8)

        rospy.loginfo("The feature database has been loaded!")

    def cloud_callback(self, msg):
        """订阅点云，并进行体素降采样"""
        self.current_cloud = self.pointcloud2_to_o3d(msg)

        # 对点云进行体素降采样
        self.downsampled_cloud = self.current_cloud.voxel_down_sample(self.voxel_size)
        rospy.loginfo(
            f"Downsampled point cloud from {len(self.current_cloud.points)} to {len(self.downsampled_cloud.points)} points"
        )

    def optimize_pose(self, pose_msg):
        """使用 ICP 进行位姿优化"""
        if self.downsampled_cloud is None:
            rospy.logwarn("No downsampled point cloud available, skipping ICP")
            return

        # 计算初始位姿变换矩阵
        T_init = self.pose_to_transformation(pose_msg)

        # 变换降采样后的点云
        transformed_cloud = self.downsampled_cloud.transform(T_init)

        # 执行 ICP（仅对降采样点云）
        icp_result = o3d.pipelines.registration.registration_icp(
            transformed_cloud,
            self.global_map,
            max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=25
            ),
        )

        # 计算最终优化的绝对位姿
        optimized_T = np.dot(T_init, icp_result.transformation)

        # 发布优化后的位姿
        opt_pose_msg = self.transformation_to_pose(optimized_T)
        self.opt_pose_publisher.publish(opt_pose_msg)
        rospy.loginfo("Published optimized pose!")

        # **每 5 秒保存一次优化后的点云**
        current_time = time.time()
        if current_time - self.last_save_time >= 5.0:
            timestamp = int(current_time)
            filename = f"optimized_cloud_{timestamp}.pcd"
            optimized_cloud = self.current_cloud.transform(optimized_T)
            o3d.io.write_point_cloud(filename, optimized_cloud)
            rospy.loginfo(f"Saved transformed point cloud to {filename}")
            self.last_save_time = current_time  # 更新最后保存时间

    def pose_to_transformation(self, pose_msg):
        """将 PoseStamped 转换为 4x4 变换矩阵"""
        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation
        T = np.eye(4)
        R = tf.quaternions.quat2mat(
            [ori.w, ori.x, ori.y, ori.z]
        )  # 四元数格式 (w, x, y, z)
        T[:3, :3] = R
        T[:3, 3] = [pos.x, pos.y, pos.z]
        return T

    def transformation_to_pose(self, T):
        """将 4x4 变换矩阵转换为 PoseStamped"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = (
            T[:3, 3]
        )
        # 计算旋转四元数
        quat = tf.quaternions.mat2quat(T[:3, :3])
        pose_msg.pose.orientation.w = quat[0]
        pose_msg.pose.orientation.x = quat[1]
        pose_msg.pose.orientation.y = quat[2]
        pose_msg.pose.orientation.z = quat[3]
        return pose_msg

    def pointcloud2_to_o3d(self, cloud_msg):
        """将 ROS PointCloud2 转换为 Open3D 点云格式"""
        points = np.array(
            [
                list(p[:3])
                for p in pc2.read_points(
                    cloud_msg, field_names=("x", "y", "z"), skip_nans=True
                )
            ]
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud

    def load_feature_database(self):
        with open(self.feature_db_path, "rb") as f:
            self.features_database = pickle.load(f)

        for data in self.features_database:
            if data["descriptors"] is not None:
                data["descriptors"] = np.array(data["descriptors"], dtype=np.uint8)

        rospy.loginfo("The feature database has been loaded!")

    def start(self):
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.loginfo("Start subscribing to the image topic...")
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("feature_matcher_node")
    matcher = FeatureMatcher(
        feature_db_path="features_database.pkl",
        global_pc_path="global_map.pcd",
        image_topic="/ricoh_camera/image_raw",
        pose_topic="/matched_pose",
        cloud_topic="/cloud_registered_body",
        opt_pose_topic="/opt_match_pose",
        voxel_size=1.0,  # 设置体素降采样大小
        visualize=False,
        width_crop=150,
        icp_opt=True
    )
    matcher.start()
