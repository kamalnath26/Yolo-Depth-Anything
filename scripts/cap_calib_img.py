'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

This File contains the code for capturing and saving calibration images from a camera
'''

import cv2
import os

SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(2)  # Use appropriate camera index

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Calibration Frame", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):  # Press 's' to save
        fname = f"{SAVE_DIR}/img_{count:02d}.jpg"
        cv2.imwrite(fname, frame)
        print(f"[INFO] Saved: {fname}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
