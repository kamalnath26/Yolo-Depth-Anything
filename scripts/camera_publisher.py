'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This File contains the implementation of a ROS2 camera publisher that captures and publishes images from a camera
'''

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # Create publisher for raw image topic
        # self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.publisher_ = self.create_publisher(Image, 'image', 10)

        # Create OpenCV capture (0 = default camera)
        self.cap = cv2.VideoCapture(2)

        # Bridge to convert OpenCV images to ROS2 sensor_msgs/Image
        self.bridge = CvBridge()

        # Timer to publish frames at ~30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Convert BGR to ROS2 image message
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Publish the image
        self.publisher_.publish(image_msg)
        self.get_logger().info('Published image frame')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
