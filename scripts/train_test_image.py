'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This script test the model with trained data with an sample image
'''

import torch
import cv2

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/kamal/prcv_ws/src/yolov5/runs/train/yolo_pose_all/weights/best.pt')

# Load test image
img = cv2.imread('/home/kamal/prcv_ws/src/prcv_project/dataset_objects_raw/blue_box_01.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
results = model(img_rgb)

# Extract and print results
results.print()
results.show()  # Opens window
results.save()  # Saves results to 'runs/detect'

# Get coordinates
for *xyxy, conf, cls in results.xyxy[0]:
    x1, y1, x2, y2 = map(int, xyxy)
    print(f"Detected object: Class {int(cls)} | Confidence {conf:.2f}")
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
