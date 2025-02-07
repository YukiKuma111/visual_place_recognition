import rospy
import cv2
import numpy as np
import open3d as o3d
import pickle
import threading
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros


class FeatureAndPointCloudProcessor:
    def __init__(self, camera_topic, pointcloud_topic, tf_topic, save_interval=1.0, voxel_size=1.0):
        # ROS topics
        self.camera_topic = camera_topic
        self.pointcloud_topic = pointcloud_topic
        self.tf_topic = tf_topic

        # Tools and parameters
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(
            nfeatures=1500, scaleFactor=1.1, nlevels=12, edgeThreshold=15
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Data storage
        self.features_database = []
        self.global_cloud = o3d.geometry.PointCloud()
        self.save_interval = save_interval
        self.voxel_size = voxel_size
        self.last_save_time = rospy.Time.now().to_sec()
        self.last_update_time = time.time()

        # Distance tracking
        self.distances = []
        self.last_pose = None
        self.recommend_interval = (
            5  # Interval for printing recommended distance (seconds)
        )

        # Thread for monitoring data timeout
        self.monitor_thread = threading.Thread(target=self.monitor_timeout)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Thread for printing recommended distance
        self.recommend_thread = threading.Thread(target=self.print_recommended_distance)
        self.recommend_thread.daemon = True
        self.recommend_thread.start()

        # Register exit handler
        rospy.on_shutdown(self.save_data)

    def image_callback(self, msg):
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_save_time < self.save_interval:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)

            if descriptors is not None:
                pose = self.get_current_pose()
                if pose is not None:
                    self.features_database.append(
                        {
                            "timestamp": msg.header.stamp.to_sec(),
                            "keypoints": [kp.pt for kp in keypoints],
                            "descriptors": descriptors.tolist(),
                            "pose": pose,  # Store pose with features
                        }
                    )

                    # 计算距离（仅 x, y 方向）
                    if self.last_pose is not None:
                        prev_x, prev_y = self.last_pose["translation"][:2]
                        curr_x, curr_y = pose["translation"][:2]
                        distance = np.sqrt(
                            (curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2
                        )
                        self.distances.append(distance)

                    # 更新上一次的位姿
                    self.last_pose = pose

                self.last_save_time = current_time
                rospy.loginfo(f"Saved features at time {msg.header.stamp.to_sec()}")

        except Exception as e:
            rospy.logwarn(f"Failed to process image: {e}")

    def get_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "camera_init", "body", rospy.Time(0), rospy.Duration(1.0)
            )
            pose = {
                "translation": [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ],
                "rotation": [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ],
            }
            return pose
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get current pose: {e}")
            return None

    def pointcloud_callback(self, msg):
        self.last_update_time = time.time()
        try:
            transform = self.tf_buffer.lookup_transform(
                "camera_init",
                msg.header.frame_id,
                msg.header.stamp,
                rospy.Duration(1.0),
            )
            transformed_cloud = do_transform_cloud(msg, transform)
            points = [
                point[:3]
                for point in pc2.read_points(transformed_cloud, skip_nans=True)
            ]
            pcl_cloud = o3d.geometry.PointCloud()
            pcl_cloud.points = o3d.utility.Vector3dVector(
                np.array(points, dtype=np.float32)
            )
            filtered_cloud = pcl_cloud.voxel_down_sample(voxel_size=self.voxel_size)
            self.global_cloud += filtered_cloud

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")

    def monitor_timeout(self):
        while not rospy.is_shutdown():
            if time.time() - self.last_update_time > 5:
                rospy.logwarn(
                    "No data received for 5 seconds. Saving data and shutting down..."
                )
                self.save_data()
                rospy.signal_shutdown("Timeout detected")
            time.sleep(1)

    def save_data(self):
        try:
            with open("features_database.pkl", "wb") as f:
                pickle.dump(self.features_database, f)
            rospy.loginfo("Feature database saved to features_database.pkl")

            o3d.io.write_point_cloud("global_map.pcd", self.global_cloud)
            rospy.loginfo("Global point cloud saved to global_map.pcd")

        except Exception as e:
            rospy.logerr(f"Failed to save data: {e}")

    def print_recommended_distance(self):
        while not rospy.is_shutdown():
            rospy.sleep(self.recommend_interval)
            if len(self.distances) >= 10:
                sorted_distances = sorted(self.distances)
                n = len(sorted_distances)
                filtered_distances = sorted_distances[n // 10 : -n // 10]
                if filtered_distances:
                    average_distance = sum(filtered_distances) / len(filtered_distances)
                    rospy.loginfo(
                        f"Recommended average distance: {average_distance:.2f} meters"
                    )

    def start(self):
        rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback)
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("feature_and_pointcloud_processor_node")
    processor = FeatureAndPointCloudProcessor(
        camera_topic="/ricoh_camera/image_raw",
        pointcloud_topic="/cloud_registered_body",
        tf_topic="/tf",
        save_interval=1.0,
        voxel_size=2.0,
    )
    processor.start()
