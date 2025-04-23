# Yolo-Depth-Anything
This is project for pose detection of object using Yolo with RGB and RGBD camera

To capture chessboard or tag for calibation use 

```bash
pyhton3 cap_calib_img.py
```

To get the camera calibration images and store calibration mat for rgb camera use 

```bash
python3 calib_cam.py
```
Capture object using 
```bash
python3 cap_objects.py
```

Use LabelImg to lable the data as given below folder structure:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

example data.yaml

```
train: /absolute/path/to/dataset/images/train
val: /absolute/path/to/dataset/images/val

nc: 1  # number of classes
names: ['my_object']
```

To splits data sets for training use:


```bash 
python3 split_dataset_train.py
```


First, clone and install YOLOv5:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```


To train the YOLOv5 model, run the following command:
```bash 
python3 train.py   --img 640   --batch 16   --epochs 100   --data /home/kamal/prcv_ws/src/prcv_project/dataset/data.yaml   --weights yolov5s.pt   --name yolo_my_object
```

--img 640: image size

--batch 16: adjust based on GPU

--weights yolov5s.pt: pretrained model

--name: output folder



To test the trained model:

make sure you have training with at least 100 epochs and at least 100 training images

```bash
python3 train_test_image.py
```

Test Trained model on live feed:

```bash
python3 train_test_live.py
```

After training, you can use the following command to run detection on images:
```bash
python3 detect.py --weights runs/train/yolo_pose_all/weights/best.pt --source /home/kamal/prcv_ws/src/prcv_project/dataset_objects_raw/blue_box_00.jpg --conf 0.1
```

After validation, you can evaluate the model's performance using the following command:
```bash
python3 val.py --weights runs/train/yolo_pose_all/weights/best.pt --data /home/kamal/prcv_ws/src/prcv_project/dataset_objects_raw/blue_box_00.jpg --img 640
```


To use pretained model to detect multiple objects and print pose in terminal on live feed using the following command:

```bash
python3 live_pose_yolo_rgb_mul.py
```

To use pretained model with ROS2 to detect multiple objects and publish pose and print pose in terminal and visualize tf in rviz use:

makes sure topics are correctly specified in the file

```bash
python3 ros2_yolo_rbg.py
```

config file for rviz is in rviz config `camera_config_rgb.rviz` and run `rviz2` and open the config

To use pretained model with ROS2 with depth data to detect multiple objects and publish pose and print pose in terminal and visualize tf in rviz use:

first launch any depth camera launch files to get data 

example:

```bash
ros2 launch realsense2_camera rs_launch.py
```
for intel realsense

```bash
ros2 launch orbbec_camera dabai_dcw.launch.py
```
for orbbec dabai dwc camera


makes sure topics are correctly specified in the file

```bash
python3 ros2_yolo_rbgd.py
```

config file for rviz is in rviz config `camera_config_rgbd.rviz` and run `rviz2` and open the config

---

## Initial testings for quick results

find object 2d commands with rgb data

```bash
ros2 launch find_object_2d find_object_2d.launch.py image_topic:=/camera/color/image_raw
```

To visualize the detected objects, you can use the following command:
```bash
ros2 run find_object_2d print_objects_detected --ros-args -r image_topic:=/camera/color/image_raw
```

find obhject 2d with depth data
```bash
ros2 launch find_object_2d find_object_3d.launch.py    rgb_topic:=/camera/color/image_raw    depth_topic:=/camera/depth/image_raw    camera_info_topic:=/camera/color/camera_info
```

To transform the detected objects into a TF (transform) format, use the following command: and print objectpos  x,y,z qat x,y,z,w
```bash
ros2 run find_object_2d tf_example
```



