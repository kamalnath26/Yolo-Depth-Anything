'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This script implements a real-time object detection system using YOLOv5 and camera calibration for accurate 3D positioning of detected objects
and integrated with ROS2 with RDG data to publish the detected object poses as ROS2 messages and visualzile the TF in Rviz2
'''


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')

        # Load YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.target_labels = ['cup', 'cell phone', 'book']
        # self.target_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        # 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        # 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        # 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        # 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        # 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        # 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
        # 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        # 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        # 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        # 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        # 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        # 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Load camera calibration
        calib = np.load("calibration_data_usb.npz")
        # calib = np.load("calibration_data_orbbec.npz")
        self.K = calib["K"]
        self.dist = calib["dist"]

        # Define object model (cuboid)
        # self.W, self.H, self.D = 0.07, 0.2, 0.07
        # self.object_points = np.array([
        #     [0, 0, 0], [self.W, 0, 0], [self.W, self.H, 0], [0, self.H, 0],
        #     [0, 0, -self.D], [self.W, 0, -self.D], [self.W, self.H, -self.D], [0, self.H, -self.D]
        # ], dtype=np.float32)
        # Define real-world dimensions (W, H, D) in meters
        self.object_dimensions = {
            'cup': (0.06, 0.10, 0.06),              # diameter, height
            'cell phone': (0.074, 0.159, 0.008),    # width, height, thickness
            'book': (0.15, 0.20, 0.01)              # width, height, thickness
        }

        self.br = CvBridge()
        
        self.create_subscription(Image, '/image', self.image_callback, 10)
        # self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)

        # Create publishers for each object class
        # self.pose_publishers = {
        #     label: self.create_publisher(PoseStamped, f'/object_pose/{label}', 10)
        #     for label in self.target_labels
        # }
        self.pose_publishers = {
            label: self.create_publisher(
                PoseStamped,
                f"/object_pose/{label.replace(' ', '_')}",
                10
            )
            for label in self.target_labels
        }

        self.image_pub = self.create_publisher(Image, '/camera/image_axes_overlay', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Pose estimator node started")

    def get_object_points(self, label):
        W, H, D = self.object_dimensions[label]
        return np.array([
            [0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0],
            [0, 0, -D], [W, 0, -D], [W, H, -D], [0, H, -D]
        ], dtype=np.float32)

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)
        detections = results.xyxy[0]

        for *xyxy, conf, cls in detections:
            label = self.model.names[int(cls)]
            if label not in self.target_labels:
                continue

            x1, y1, x2, y2 = map(int, xyxy)

            image_points = np.array([
                [x1, y2], [x2, y2], [x2, y1], [x1, y1],
                [x1 + 10, y2 + 10], [x2 - 10, y2 + 10],
                [x2 - 10, y1 - 10], [x1 + 10, y1 - 10]
            ], dtype=np.float32)

            # success, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.K, self.dist)
            object_points = self.get_object_points(label)
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, self.K, self.dist)

            if not success:
                continue

            rot_mat, _ = cv2.Rodrigues(rvec)
            quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_frame"

            pose_msg.pose.position.x = tvec[0][0]
            pose_msg.pose.position.y = tvec[1][0]
            pose_msg.pose.position.z = tvec[2][0]

            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.pose_publishers[label].publish(pose_msg)
            # self.get_logger().info(f"[{label.upper()}] Pose published")

            ros_topic_name = f"/object_pose/{label.replace(' ', '_')}"
            self.get_logger().info(f"[{label.upper()}] Pose published → {ros_topic_name}")
            # Log the PoseStamped message
            self.get_logger().info(
                f"[{label.upper()}] PoseStamped:\n"
                f"  Header:\n"
                f"    frame_id: {pose_msg.header.frame_id}\n"
                f"    stamp: {pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}\n"
                f"  Position:\n"
                f"    x: {pose_msg.pose.position.x:.3f}, "
                f"y: {pose_msg.pose.position.y:.3f}, "
                f"z: {pose_msg.pose.position.z:.3f}\n"
                f"  Orientation (quaternion):\n"
                f"    x: {pose_msg.pose.orientation.x:.4f}, "
                f"y: {pose_msg.pose.orientation.y:.4f}, "
                f"z: {pose_msg.pose.orientation.z:.4f}, "
                f"w: {pose_msg.pose.orientation.w:.4f}"
            )

            # Draw 3D pose axes
            frame = cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, 0.05)

            # Draw label and XYZ text
            cv2.putText(
                frame,
                f"{label} XYZ: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"{label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # to view /pose_overlay_image in rqt_image_view
            # Publish processed image with pose axes
            self.image_pub.publish(self.br.cv2_to_imgmsg(frame, encoding='bgr8'))

            # Broadcast TF
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_frame'
            t.child_frame_id = f'{label.replace(" ", "_")}_pose'

            t.transform.translation.x = tvec[0][0]
            t.transform.translation.y = tvec[1][0]
            t.transform.translation.z = tvec[2][0]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(t)




def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()