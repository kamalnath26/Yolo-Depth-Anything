'''
Kamalnath Bathirappan & Ahilesh Rajaram
April 2025
CS 5330 Project Final

Dataset Splitter for Image Classification for Yolo Training
'''

import os
import shutil
import random

base = "/home/kamal/prcv_ws/src/prcv_project/dataset"
image_dir = os.path.join(base, "images/train")
label_dir = os.path.join(base, "labels/train")

val_image_dir = os.path.join(base, "images/val")
val_label_dir = os.path.join(base, "labels/val")
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Split 20% to val
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
val_size = int(0.2 * len(image_files))
val_files = random.sample(image_files, val_size)

for file in val_files:
    img_path = os.path.join(image_dir, file)
    label_path = os.path.join(label_dir, file.replace('.jpg', '.txt').replace('.png', '.txt'))
    
    shutil.move(img_path, os.path.join(val_image_dir, file))
    shutil.move(label_path, os.path.join(val_label_dir, os.path.basename(label_path)))

print(f"Moved {val_size} image-label pairs to validation set.")
