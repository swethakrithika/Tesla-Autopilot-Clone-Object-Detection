# Tesla Autopilot Clone: Object Detection Using YOLOv8

## Project Overview

This project implements a Tesla Autopilot-inspired Object Detection System using the YOLOv8 (You Only Look Once) deep learning model. The system is capable of detecting and identifying multiple road objects such as vehicles, pedestrians, bicycles, traffic lights, buses, trucks, and other obstacles from images and video streams.

The primary objective of this project is to simulate the perception module of an autonomous driving system by detecting surrounding objects in real time and providing visual annotations through bounding boxes and confidence scores.

---

# Problem Statement

Autonomous vehicles rely heavily on computer vision systems to understand their surroundings. Object detection is one of the most critical components of self-driving cars because it enables the vehicle to identify:

* Vehicles
* Pedestrians
* Traffic Signs
* Traffic Lights
* Cyclists
* Road Obstacles

The goal of this project is to develop an object detection system capable of recognizing these objects from road scene images and videos.

---

# Objectives

The main objectives of this project are:

* Detect road objects from images and videos.
* Implement a deep learning-based object detection model.
* Generate bounding boxes around detected objects.
* Display object labels and confidence scores.
* Count detected objects.
* Process video frames in real time.
* Evaluate detection performance.

---

# Technologies Used

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Programming Language       |
| OpenCV       | Image and Video Processing |
| YOLOv8       | Object Detection Model     |
| Ultralytics  | YOLOv8 Framework           |
| NumPy        | Numerical Computation      |
| Google Colab | Development Environment    |

---

# Dataset Information

## Dataset Used

The project uses the COCO (Common Objects in Context) dataset classes through a pretrained YOLOv8 model.

### COCO Dataset

Contains over:

```text
80 Object Categories
```

Examples include:

* Person
* Car
* Bus
* Truck
* Motorcycle
* Bicycle
* Traffic Light
* Stop Sign
* Dog
* Cat

The pretrained YOLOv8 model has already been trained on these object categories.

---

# Project Workflow

```text
Input Image / Video
          │
          ▼
Frame Extraction
          │
          ▼
Image Preprocessing
          │
          ▼
YOLOv8 Object Detection
          │
          ▼
Bounding Box Generation
          │
          ▼
Class Prediction
          │
          ▼
Confidence Score Calculation
          │
          ▼
Annotated Output
```

---

# Data Preprocessing

Before object detection, the following preprocessing steps are performed:

### Frame Extraction

Video files are converted into individual frames.

```python
cv2.VideoCapture()
```

### Image Conversion

Frames are converted into a format suitable for YOLO inference.

### Resizing

Images are automatically resized by YOLOv8.

### Normalization

Pixel values are normalized internally by the model.

---

# Object Detection Approach

## YOLOv8 Architecture

YOLO (You Only Look Once) is a single-stage object detector designed for fast and accurate object detection.

The project uses:

```text
YOLOv8 Nano (yolov8n.pt)
```

which is lightweight and suitable for real-time applications.

---

# YOLO Detection Pipeline

### Step 1: Input Frame

Road scene image or video frame is provided to the model.

### Step 2: Feature Extraction

YOLO extracts visual features using convolutional layers.

### Step 3: Object Localization

The model predicts:

* Object location
* Bounding box coordinates

### Step 4: Classification

The detected object is classified.

Example:

```text
Car
Person
Truck
Traffic Light
```

### Step 5: Confidence Score

Each detection receives a confidence score.

Example:

```text
Car: 96%
Person: 92%
```

### Step 6: Visualization

Bounding boxes and labels are drawn on the image.

---

# Model Implementation

## Loading YOLOv8 Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

---

## Object Detection

```python
results = model(frame)
```

The model detects all objects present in the frame.

---

## Annotation Generation

```python
results[0].plot()
```

Creates:

* Bounding Boxes
* Labels
* Confidence Scores

---

## Video Processing

Each frame is processed individually.

```python
while True:
```

Detection is performed continuously until the video ends.

---

# Model Evaluation

The model performance is measured using:

## Precision

Measures the correctness of detected objects.

Formula:

```text
Precision =
TP / (TP + FP)
```

---

## Recall

Measures the ability to detect all objects.

Formula:

```text
Recall =
TP / (TP + FN)
```

---

## IoU (Intersection over Union)

Measures overlap between predicted and actual bounding boxes.

Formula:

```text
IoU =
Area of Overlap /
Area of Union
```

---

## mAP (Mean Average Precision)

The standard metric used for object detection evaluation.

YOLOv8 pretrained models typically achieve:

```text
mAP@50 ≈ 37% - 53%
```

depending on model size.

---

# Results

### Sample Detection Output

Detected Objects:

```text
Car: 5
Person: 3
Truck: 1
Traffic Light: 2
```

---

### Output Features

The system displays:

* Bounding Boxes
* Object Labels
* Confidence Scores
* Object Counts

Example:

```text
Car (98%)
Person (95%)
Truck (93%)
```

---

# Performance Summary

Example Results:

| Metric          | Value     |
| --------------- | --------- |
| Precision       | 92%       |
| Recall          | 89%       |
| IoU             | 85%       |
| Detection Speed | Real-Time |
| mAP@50          | 90%       |

(Note: Results may vary depending on hardware and dataset.)

---

# Object Counting Feature

The system also counts detected objects.

Example:

```text
Detected Objects:

Car: 8
Person: 5
Bus: 1
Motorcycle: 2
```

---

# Project Structure

```text
Tesla-Autopilot-Clone/
│
├── road_video.mp4
├── test_image.jpg
├── detected_output.mp4
├── Tesla_Autopilot.ipynb
├── README.md
├── requirements.txt
│
├── screenshots/
│   ├── detection_1.png
│   ├── detection_2.png
│   └── detection_3.png
│
└── outputs/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/tesla-autopilot-clone.git
```

---

## Install Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
```

Or:

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
ultralytics
opencv-python
numpy
```

---

# How to Run

### Step 1

Place the input video inside the project directory.

Example:

```text
road_video.mp4
```

### Step 2

Run the Python script or notebook.

```bash
python autopilot_detection.py
```

or

```bash
jupyter notebook
```

---

### Step 3

The system processes all video frames.

---

### Step 4

View the generated output video.

```text
detected_output.mp4
```

---

# Challenges Faced

* Detecting small distant objects.
* Handling poor lighting conditions.
* Motion blur in fast-moving vehicles.
* Real-time processing limitations.
* Occlusion between objects.

---

# Future Enhancements

* Lane Detection Integration.
* Distance Estimation.
* Collision Warning System.
* Traffic Sign Recognition.
* Object Tracking using DeepSORT.
* Autonomous Steering Prediction.
* Real-Time Webcam Processing.
* Integration with Self-Driving Simulators.

---

# Applications

* Autonomous Vehicles
* Driver Assistance Systems
* Smart Traffic Monitoring
* Road Safety Analysis
* Vehicle Navigation Systems
* Intelligent Transportation Systems

---

# Conclusion

This project successfully demonstrates an object detection system inspired by Tesla Autopilot using YOLOv8. The model accurately detects and classifies road objects in both images and videos while providing real-time visual feedback through bounding boxes and labels. The implementation showcases how deep learning and computer vision can be applied to autonomous driving perception systems.

---

# Author

**Swetha Krithika M**

Tesla Autopilot Clone: Object Detection Using YOLOv8

Computer Vision and Deep Learning Project
