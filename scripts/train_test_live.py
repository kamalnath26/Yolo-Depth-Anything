'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This script test the model with trained data with an live feed
'''


import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/kamal/prcv_ws/src/yolov5/runs/train/yolo_pose_all/weights/best.pt')

cap = cv2.VideoCapture(2)  # Adjust camera index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
