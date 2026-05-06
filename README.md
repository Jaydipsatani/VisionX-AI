<div align="center">

<h1>VisionX AI</h1>

<p><strong>Transforming the way humans interact with machines using Computer Vision</strong></p>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-0097A7?style=for-the-badge&logo=google&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B6B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> VisionX AI is an advanced real-time computer vision system that combines three powerful AI modules — Gesture-Controlled Virtual Mouse, Face Recognition, and Object Detection — into one unified platform built entirely with Python.

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Modules](#-modules)
  - [Module 01 — AI Virtual Mouse](#-module-01--ai-virtual-mouse)
  - [Module 02 — Face Recognition](#-module-02--face-recognition)
  - [Module 03 — Object Detection](#-module-03--object-detection)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Author](#-author)

---

## Overview

VisionX AI is a modular computer vision platform that allows humans to interact with machines in smarter, more intuitive ways. Each module tackles a different real-world problem using state-of-the-art AI techniques:

| Module | Technology | Performance |
|--------|-----------|-------------|
| AI Virtual Mouse | MediaPipe + OpenCV | 20+ FPS, 21 hand landmarks |
| Face Recognition | Deep Learning + OpenCV | 96.94% accuracy |
| Object Detection | YOLOv8 + Ultralytics | 87%+ confidence, multi-object |

No cloud. No external API. Everything runs **100% locally** on Python.

---

## Modules

### Module 01 — AI Virtual Mouse

> Control your entire computer using just your hand — no physical mouse needed.

**How it works:**
1. Webcam captures live hand feed
2. MediaPipe detects **21 hand landmarks** in real-time
3. Index finger tip position is mapped to screen coordinates
4. Pinching index + middle finger triggers a **mouse click**
5. OpenCV handles smooth coordinate mapping and rendering

**Key Features:**
- Real-time cursor control via hand gesture
- Click, move, and navigate without any hardware
- Smooth motion using coordinate interpolation
- Runs at **20+ FPS** on standard hardware

**Files:** `AI Virtual Mouse/`

---

### Module 02 — Face Recognition

> AI that identifies who you are — live from your webcam — with 96.94% accuracy.

**How it works:**
1. OpenCV captures live webcam feed
2. Haar Cascade / Deep Learning model detects the face region
3. Trained model compares the face against known identity data
4. Outputs **name + confidence score** in real-time
5. Green bounding box drawn around the recognized face

**Key Features:**
- Real-time identity detection from webcam
- Custom-trained on personal dataset
- Achieves **96.94% confidence** on known faces
- Displays name label directly on the video feed

**Files:** `Face Recognition/`

---

### Module 03 — Object Detection

> Detect multiple objects simultaneously in live video using YOLOv8.

**How it works:**
1. Webcam feed is processed frame by frame
2. YOLOv8 model analyzes each frame for known objects
3. Bounding boxes drawn with **class label + confidence score**
4. Multiple objects detected in a single frame simultaneously

**What it can detect (examples from live demo):**
-  Person — 0.59 confidence
-  Cell Phone — 0.38 confidence
-  Motorcycle — 0.73 confidence
- *(80+ object classes supported via COCO dataset)*

**Key Features:**
- Real-time multi-object detection
- 87%+ confidence on clear objects
- Powered by **YOLOv8** (state-of-the-art detection model)
- Works on both live webcam and video files

**Files:** `Object Detection/`

---

##  Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Video capture, image processing, rendering |
| **MediaPipe** | Hand landmark detection (21 points) |
| **NumPy** | Numerical operations & coordinate mapping |
| **YOLOv8 (Ultralytics)** | Real-time object detection |
| **Deep Learning (face_recognition / dlib)** | Face encoding & recognition |
| **HTML / JS** | VisionX AI web interface (`index.html`) |

---

## Project Structure

```
VisionX-AI/
│
├── AI Virtual Mouse/
│   ├── virtual_mouse.py        # Main gesture mouse script
│   └── requirements.txt        # Module dependencies
│
├── Face Recognition/
│   ├── face_recognition.py     # Main face recognition script
│   ├── training_images/        # Custom face dataset
│   └── requirements.txt
│
├── Object Detection/
│   ├── object_detection.py     # YOLOv8 detection script
│   └── requirements.txt
│
├── index.html                  # VisionX AI web portfolio
├── server.py                   # Local server for web interface
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/JaydeepSatani/VisionX-AI.git
cd VisionX-AI
```

**2. Install dependencies**
```bash
pip install opencv-python mediapipe numpy ultralytics face-recognition
```

Or install from each module's requirements file:
```bash
pip install -r "AI Virtual Mouse/requirements.txt"
pip install -r "Face Recognition/requirements.txt"
pip install -r "Object Detection/requirements.txt"
```

**Requirements:**
- Python 3.8 or higher
- Webcam (built-in or external)
- Windows / Linux / macOS

---

## How to Run

**Module 01 — AI Virtual Mouse**
```bash
cd "AI Virtual Mouse"
python virtual_mouse.py
```

**Module 02 — Face Recognition**
```bash
cd "Face Recognition"
python face_recognition.py
```

**Module 03 — Object Detection**
```bash
cd "Object Detection"
python object_detection.py
```

**Web Interface**
```bash
python server.py
# Open index.html in your browser
```

---

## Results

| Module | Metric | Value |
|--------|--------|-------|
| AI Virtual Mouse | FPS | 20–21 FPS |
| AI Virtual Mouse | Hand Landmarks | 21 points |
| Face Recognition | Accuracy | 96.94% |
| Face Recognition | Detection | Real-time live |
| Object Detection | Confidence | 73–87%+ |
| Object Detection | Objects/Frame | Multiple simultaneously |

---

## Author

**Jaydeep Satani**  
AI & Computer Vision Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Jaydeep_Satani-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jaydeep-satani)
[![GitHub](https://img.shields.io/badge/GitHub-JaydeepSatani-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/JaydeepSatani)

---

<div align="center">

 **If you found this project helpful, please give it a star!** 

*Built with Python & Computer Vision*

</div>
