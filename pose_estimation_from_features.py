import pickle
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


class FeatureMatcher:
    def __init__(self, feature_db_path, image_topic):
        self.feature_db_path = feature_db_path
        self.image_topic = image_topic
        self.features_database = []
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.pose_publisher = rospy.Publisher(
            "/matched_pose", PoseStamped, queue_size=10
        )
        self.load_feature_database()

    def load_feature_database(self):
        # 加载特征数据
        with open(self.feature_db_path, "rb") as f:
            self.features_database = pickle.load(f)

        # 转换描述子为 numpy 数组
        for data in self.features_database:
            if data["descriptors"] is not None:
                data["descriptors"] = np.array(data["descriptors"], dtype=np.uint8)

        rospy.loginfo(
            "The feature database has been loaded and converted into a format suitable for OpenCV!"
        )

    def image_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像
            new_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

            # 提取实时图像特征
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)

            if descriptors is None:
                rospy.logwarn(
                    "No features were extracted from the live image. Please check the image quality!"
                )
                return

            # 进行匹配
            best_match = None
            min_distance = float("inf")

            for data in self.features_database:
                if data["descriptors"] is None:
                    continue

                # 匹配描述子
                matches = self.bf.match(descriptors, data["descriptors"])

                # 计算平均匹配距离
                avg_distance = sum([m.distance for m in matches]) / len(matches)

                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_match = data

            if best_match:
                rospy.loginfo(
                    f"Best matching image timestamp: {best_match['timestamp']}"
                )
                rospy.loginfo(f"Pose: {best_match['pose']}")

                # 发布 Pose 信息
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = best_match["pose"]["translation"][0]
                pose_msg.pose.position.y = best_match["pose"]["translation"][1]
                pose_msg.pose.position.z = best_match["pose"]["translation"][2]
                pose_msg.pose.orientation.x = best_match["pose"]["rotation"][0]
                pose_msg.pose.orientation.y = best_match["pose"]["rotation"][1]
                pose_msg.pose.orientation.z = best_match["pose"]["rotation"][2]
                pose_msg.pose.orientation.w = best_match["pose"]["rotation"][3]
                self.pose_publisher.publish(pose_msg)
                rospy.loginfo("Publish matching Pose information!")

                # 可视化关键点
                curr_image = cv2.drawKeypoints(
                    new_image, keypoints, None, color=(0, 255, 0)
                )
                cv2.imshow("Keypoints for current picture", curr_image)
                cv2.waitKey(1)
            else:
                rospy.loginfo("No suitable matches found!")

        except Exception as e:
            rospy.logerr(f"Error while processing image: {e}")

    def start(self):
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.loginfo("Start subscribing to the image topic...")
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("feature_matcher_node")

    # 初始化 FeatureMatcher
    matcher = FeatureMatcher(
        feature_db_path="features_database.pkl", image_topic="/usb_cam/image_raw"
    )

    # 启动匹配流程
    matcher.start()
