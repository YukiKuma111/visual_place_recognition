#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publish_image():
    # 初始化ROS节点
    rospy.init_node('image_publisher', anonymous=True)
    
    # 创建一个发布者，发布到 /usb_cam/image_raw
    image_pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=10)
    
    # 使用 CvBridge 将OpenCV图像转换为ROS图像消息
    bridge = CvBridge()
    
    # 加载PNG图片
    image_path = "xxx.png"  # 替换为您的PNG图片路径
    cv_image = cv2.imread(image_path)
    
    if cv_image is None:
        rospy.logerr("Unable to load image at path: {}".format(image_path))
        return
    
    # 设置发布频率
    rate = rospy.Rate(10)  # 10Hz

    rospy.loginfo("Publishing image from: {}".format(image_path))
    while not rospy.is_shutdown():
        # 将OpenCV图像转换为ROS图像消息
        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        
        # 发布图像消息
        image_pub.publish(ros_image)
        
        # 等待
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass
