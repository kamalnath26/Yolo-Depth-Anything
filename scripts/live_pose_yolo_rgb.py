'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This script implements a real-time object detection system using YOLOv5 and camera calibration for accurate 3D positioning of detected objects.
'''

import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load camera calibration
calib = np.load("calibration_data_usb.npz")
K = calib["K"]
dist = calib["dist"]

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define target labels
target_labels = ['cup', 'cell phone', 'book']

# target_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
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

# Define object models (W, H, D in meters)
object_dimensions = {
    'cup': (0.06, 0.10, 0.06),           # diameter=6cm, height=10cm
    'cell phone': (0.074, 0.159, 0.008), # Samsung S20 FE: 7.4cm x 15.9cm x 0.8cm
    'book': (0.15, 0.20, 0.01)           # Small notebook: 15cm x 20cm x 1cm
}

def get_object_points(label):
    W, H, D = object_dimensions[label]
    return np.array([
        [0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0],
        [0, 0, -D], [W, 0, -D], [W, H, -D], [0, H, -D]
    ], dtype=np.float32)

# Start video stream
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    for *xyxy, conf, cls in detections:
        label = model.names[int(cls)]
        if label not in target_labels:
            continue

        x1, y1, x2, y2 = map(int, xyxy)

        # Image points from bounding box corners (approximate)
        image_points = np.array([
            [x1, y2], [x2, y2], [x2, y1], [x1, y1],
            [x1 + 10, y2 + 10], [x2 - 10, y2 + 10],
            [x2 - 10, y1 - 10], [x1 + 10, y1 - 10]
        ], dtype=np.float32)

        object_points = get_object_points(label)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist)
        if success:
            # Draw pose
            frame = cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.05)

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()

            print(f"\n[DETECTED] {label.upper()}")
            print(f"  Position (x, y, z): {tvec.flatten()}")
            print(f"  Orientation (Rodrigues): {rvec.flatten()}")
            print(f"  Orientation (Quaternion): {quat}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Live Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
