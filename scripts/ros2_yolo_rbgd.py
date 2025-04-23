'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This script implements a real-time object detection system using YOLOv5 and camera calibration for accurate 3D positioning of detected objects
and integrated with ROS2 with RDGD data to publish the detected object poses as ROS2 messages and visualzile the TF in Rviz2
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from message_filters import ApproximateTimeSynchronizer, Subscriber


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PoseEstimatorRGBDNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_rgbd_node')

        # Load YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.target_labels = ['cup', 'cell phone', 'book']

        # Load camera calibration
        calib = np.load("calibration_data_orbbec.npz")
        self.K = calib["K"]
        self.dist = calib["dist"]

        # Define 3D cuboid model (approx in meters)
        # self.W, self.H, self.D = 0.07, 0.2, 0.07
        # self.object_points = np.array([
        #     [0, 0, 0], [self.W, 0, 0], [self.W, self.H, 0], [0, self.H, 0],
        #     [0, 0, -self.D], [self.W, 0, -self.D], [self.W, self.H, -self.D], [0, self.H, -self.D]
        # ], dtype=np.float32)

        # Real-world object dimensions in meters
        self.object_dimensions = {
            'cup': (0.06, 0.10, 0.06),              # diameter, height, same depth
            'cell phone': (0.074, 0.159, 0.008),    # width, height, thickness
            'book': (0.15, 0.20, 0.01)              # width, height, thickness
}

        # Subscribers (RGB + Depth sync)
        self.bridge = CvBridge()
        rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.rgbd_callback)

        # Publishers
        self.pose_publishers = {
            label: self.create_publisher(PoseStamped, f'/object_pose/{label.replace(" ", "_")}', 10)
            for label in self.target_labels
        }
        self.image_pub = self.create_publisher(Image, '/camera/image_axes_overlay', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Pose estimator node (RGB + Depth) started")

    def get_object_points(self, label):
        W, H, D = self.object_dimensions[label]
        return np.array([
            [0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0],
            [0, 0, -D], [W, 0, -D], [W, H, -D], [0, H, -D]
        ], dtype=np.float32)

    def rgbd_callback(self, rgb_msg, depth_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        results = self.model(frame)
        detections = results.xyxy[0]

        for *xyxy, conf, cls in detections:
            label = self.model.names[int(cls)]
            if label not in self.target_labels:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # # Depth value at the center (assumed in meters; convert if in mm)
            # z = float(depth_image[cy, cx])
            # if z == 0.0 or np.isnan(z):
            #     self.get_logger().warn(f"No depth at {label} center point")
            #     continue
            # self.get_logger().info(f"[{label.upper()}] Depth at center (cx={cx}, cy={cy}): {z:.4f} meters")

            # Extract the depth ROI inside bounding box
            depth_roi = depth_image[y1:y2, x1:x2]

            # Mask out invalid depths (zeros or NaNs)
            valid_depths = depth_roi[(depth_roi > 0.0) & ~np.isnan(depth_roi)]

            # Check if there are any valid depth pixels
            if valid_depths.size == 0:
                self.get_logger().warn(f"[{label.upper()}] No valid depth values in bounding box")
                continue

            # Compute average Z value in meters
            z = float(np.mean(valid_depths))
            self.get_logger().info(f"[{label.upper()}] Average depth in bounding box: {z:.4f} meters")


            # Adjust 3D model scale with depth
            # scale = z / 0.5  # adjust reference distance
            scale = z / 5  # adjust reference distance
            # object_points_scaled = self.object_points * scale
            object_points = self.get_object_points(label)
            object_points_scaled = object_points * scale


            # Fake 2D keypoints from bbox corners
            image_points = np.array([
                [x1, y2], [x2, y2], [x2, y1], [x1, y1],
                [x1 + 10, y2 + 10], [x2 - 10, y2 + 10],
                [x2 - 10, y1 - 10], [x1 + 10, y1 - 10]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(object_points_scaled, image_points, self.K, self.dist)
            if not success:
                continue

            # Convert to quaternion
            rot_mat, _ = cv2.Rodrigues(rvec)
            quat = R.from_matrix(rot_mat).as_quat()

            # Publish PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            # pose_msg.header.frame_id = "camera_frame"
            pose_msg.header.frame_id = "camera_link"
            pose_msg.pose.position.x = tvec[0][0]
            pose_msg.pose.position.y = tvec[1][0]
            pose_msg.pose.position.z = tvec[2][0]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.pose_publishers[label].publish(pose_msg)

            # Log published PoseStamped
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

            # Broadcast TF
            t = TransformStamped()
            t.header = pose_msg.header
            t.child_frame_id = f"{label.replace(' ', '_')}_pose"
            t.transform.translation.x = pose_msg.pose.position.x
            t.transform.translation.y = pose_msg.pose.position.y
            t.transform.translation.z = pose_msg.pose.position.z
            t.transform.rotation = pose_msg.pose.orientation
            self.tf_broadcaster.sendTransform(t)

            # Draw on image
            frame = cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, 0.05)
            cv2.putText(frame, f"{label} XYZ: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})",
                        (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # Publish image with pose overlay
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorRGBDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
