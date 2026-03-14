# Real-Time Object Detection with YOLO and OpenCV

This project performs real-time object detection from a webcam using YOLO and OpenCV.

## Demo
assets/demo.gif

## Features
- Real-time webcam detection
- YOLOv3-tiny model for improved performance
- Frame-skipping optimization
- Non-max suppression filtering

## Technologies
- Python
- OpenCV
- YOLOv3 / YOLOv3-tiny

## Setup
1. Clone the repository
2. Install dependencies
3. Download the YOLO model weights (yolov3.weights, yolov3.cfg, coco.names)
4. Run the script


Download YOLO Model Files

This project uses pretrained YOLO models which are not included in the repository due to their size.

Download the following files:

Weights

https://pjreddie.com/media/files/yolov3-tiny.weights

Config

https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg

Class labels

https://github.com/pjreddie/darknet/blob/master/data/coco.names

## Model Files

Place them in the `models/` directory:

```
opencv-yolo-object-detection/
│
├── camera.py
├── download_models.py
├── requirements.txt
├── README.md
├── models/
├── assets/
│   └── demo.gif
└── videos/
```

Alternatively you may download the above files by running the `download_models.py` script.
They will automatically be placed in the correct location (note that these files require ~270MB of disk space).
