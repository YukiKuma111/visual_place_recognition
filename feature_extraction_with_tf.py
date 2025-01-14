import rospy
import cv2
from cv_bridge import CvBridge
import tf2_ros
import pickle
import numpy as np
from sensor_msgs.msg import Image


class FeatureExtractor:
    def __init__(
        self, camera_topic="/usb_cam/image_raw", tf_topic="/tf", save_interval=2.5
    ):
        # ROS Topics
        self.camera_topic = camera_topic
        self.tf_topic = tf_topic

        # Tools and parameters
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Feature database and saving parameters
        self.features_database = []
        self.save_interval = save_interval
        self.last_save_time = rospy.Time.now().to_sec()

        # Register exit handler to save data
        rospy.on_shutdown(self.save_to_file)

    def get_robot_pose(self):
        """Get the current robot pose from TF."""
        try:
            trans = self.tf_buffer.lookup_transform(
                "camera_init", "body", rospy.Time(0), rospy.Duration(1.0)
            )
            pose = {
                "translation": [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ],
                "rotation": [
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ],
            }
            return pose
        except Exception as e:
            rospy.logwarn(f"Failed to get transform: {e}")
            return None

    def image_callback(self, msg):
        """Process the incoming image, extract features, and store them with pose."""
        current_time = rospy.Time.now().to_sec()

        # Control saving interval
        if current_time - self.last_save_time < self.save_interval:
            return

        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"Failed to convert image: {e}")
            return

        # Convert to grayscale and extract features
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)

        # Get robot pose
        pose = self.get_robot_pose()

        if descriptors is not None and pose is not None:
            # Save features and pose
            self.features_database.append(
                {
                    "timestamp": msg.header.stamp.to_sec(),
                    "keypoints": [
                        kp.pt for kp in keypoints
                    ],  # Store only the keypoint positions
                    "descriptors": descriptors.tolist(),  # Convert descriptors to a list for serialization
                    "pose": pose,
                }
            )
            self.last_save_time = current_time
            rospy.loginfo(
                f"Saved features and pose at time {msg.header.stamp.to_sec()}"
            )

    def save_to_file(self):
        """Save the feature database to a file."""
        try:
            with open("features_database.pkl", "wb") as f:
                pickle.dump(self.features_database, f)
            rospy.loginfo("Feature database saved to file!")
        except Exception as e:
            rospy.logerr(f"Failed to save feature database: {e}")

    def start(self):
        """Start the ROS subscriber."""
        rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("feature_extraction_node")

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(
        camera_topic="/usb_cam/image_raw",
        tf_topic="/tf",
        save_interval=2.5,  # Save features frequency (unit: second)
    )

    # Start feature extraction
    feature_extractor.start()
