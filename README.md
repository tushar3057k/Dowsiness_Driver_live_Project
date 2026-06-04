# AI Driver Drowsiness and Fatigue Detection System

Real-Time Driver Fatigue and Attention Monitoring System using Computer Vision and Machine Learning.

This project is an AI-powered real-time Driver Monitoring System (DMS) developed using Python, OpenCV, MediaPipe, and Machine Learning to detect:

* Driver drowsiness
* Fatigue
* Yawning
* Eye closure
* Distraction
* Head movement
* Attention loss

The system monitors a driver's facial behavior through a webcam and detects signs of fatigue and inattention using multiple physiological indicators including eye closure, blinking behavior, yawning behavior, and head pose changes.

The system calculates a real-time Fatigue Score (%) and classifies the driver state into:

* Awake
* Sleepy
* Very Drowsy
* Distracted

---

# Overview

Driver fatigue is one of the leading causes of road accidents worldwide. Traditional fatigue monitoring systems often require specialized sensors and expensive hardware.

This project demonstrates a low-cost computer vision solution that monitors driver alertness using only a standard webcam and AI-based facial analysis.

The system performs:

* Real-time facial landmark detection
* Fatigue feature extraction
* Driver state classification
* Attention monitoring
* Alert generation
* Audible warning system

---

# Features Implemented

## Real-Time Face Detection

* MediaPipe Face Mesh based facial landmark tracking
* Real-time webcam processing
* Facial landmark visualization

---

## Eye Aspect Ratio (EAR) Detection

Detects:

* Eye closure
* Blink detection
* Blink frequency
* Eye fatigue patterns

---

## Yawn Detection

Uses Mouth Aspect Ratio (MAR) to detect:

* Yawns
* Frequent yawning events
* Mouth opening patterns

---

## Head Pose Detection

Detects:

* Head down movement
* Side distraction
* Driver inattentiveness
* Eye slope/head tilt

---

## Fatigue Score Calculation

The system calculates a weighted fatigue percentage using:

FatigueScore =

0.35E + 0.20B + 0.20M + 0.10Y + 0.10V + 0.03S + 0.02H

Where:

| Variable | Meaning                |
| -------- | ---------------------- |
| E        | Eye Closure Score      |
| B        | Blink Score            |
| M        | Mouth/Yawn Score       |
| Y        | Yawn Frequency         |
| V        | Vertical Head Ratio    |
| S        | Eye Slope              |
| H        | Horizontal Distraction |

---

## Driver State Classification

The system classifies the driver into:

| State       | Description                 |
| ----------- | --------------------------- |
| Awake       | Normal condition            |
| Sleepy      | Increased blinking/fatigue  |
| Very Drowsy | High fatigue or eye closure |
| Distracted  | Looking away or head tilt   |

---

## Real-Time Monitoring

* Dynamic fatigue updates
* Real-time blink/yawn window analysis
* Smooth fatigue transitions
* Automatic recovery from sleepy state
* Live visualization of monitoring parameters

---

## Alert System

### Visual Alerts

The system displays:

* ALERT
* DROWSINESS DETECTED
* WAKE UP

when fatigue is detected.

### Audible Alerts

An alarm sound is triggered when fatigue persists continuously for multiple frames.

This reduces false alarms caused by temporary blinks or short distractions.

---

# System Architecture

```text
Webcam Feed
      │
      ▼
MediaPipe Face Mesh
      │
      ▼
Facial Landmark Extraction
      │
      ▼
Feature Engineering
      │
      ▼
Fatigue Score Calculation
      │
      ▼
Driver State Classification
      │
      ▼
Alert and Alarm Generation
```

---

# Extracted Features

The following features are extracted in real time:

## Eye Features

* Eye Aspect Ratio (EAR)
* Blink Rate

## Mouth Features

* Mouth Aspect Ratio (MAR)
* Yawn Rate

## Head Pose Features

* Horizontal Head Deviation
* Vertical Face Ratio
* Eye Slope (Head Tilt)

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* SciPy
* Pandas
* Scikit-Learn
* Joblib

---

# Project Structure

```text
OPENMDG_DROWSINES/
│
├── app.py
├── requirements.txt
├── README.md
│
├── sound/
│   └── alarm.wav
│
├── utils/
│   ├── alarm.py
│   ├── fatigue_score.py
│   ├── driver_state.py
│   ├── eye_detection.py
│   ├── yawn_detection.py
│   ├── head_pose.py
│   ├── face_mesh.py
│   └── facedetection.py
│
├── assets/
├── models/
└── notebooks/
```

---

# How to Run

## Clone Repository

```bash
git clone <repository-url>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python app.py
```

Press:

```text
q
```

to exit the application.

---

# Current System Capabilities

* Eye Tracking
* Blink Detection
* Yawn Detection
* Fatigue Percentage
* Driver State Classification
* Head Tilt Detection
* Side Distraction Detection
* Real-Time Webcam Monitoring
* Real-Time Facial Landmark Tracking
* Visual and Audible Alerts

---

# Dataset Collection

A custom dataset can be created using the developed webcam monitoring system.

## Data Collection Procedure

Two driver states can be recorded:

### Alert State (Label = 0)

* Eyes open
* Normal blinking
* Looking straight
* No yawning

### Fatigued State (Label = 1)

* Simulated drowsiness
* Frequent yawning
* Extended eye closure
* Head tilting
* Looking down

Extracted features can be stored in CSV format for future Machine Learning training.

Example dataset format:

```csv
EAR,BlinkRate,MAR,YawnRate,VerticalRatio,EyeSlope,HorizontalDiff,Fatigue
0.31,14.2,0.24,0.0,0.44,2.1,4.3,0
0.17,31.5,0.71,4.8,0.63,15.7,12.4,1
```

---

# Machine Learning Integration

Currently the project uses a rule-based driver classification system.

Future upgrades can include Machine Learning and Deep Learning models such as:

* Logistic Regression
* Random Forest
* XGBoost
* SVM
* LSTM
* CNN-LSTM

using features:

* EAR
* MAR
* Blink Rate
* Head Pose
* Yawns
* Fatigue Score
* Eye Slope
* Distraction Ratio

Then replace:

```python
classify()
```

with:

```python
model.predict(features)
```

This will convert the system into a fully AI-powered Driver Monitoring System suitable for advanced automotive and safety applications.

---

# Future Implementations

## Advanced Features

* Night Vision Support using Histogram Equalization and Brightness Correction
* Dashboard UI using Streamlit, Tkinter, or PyQt
* Driver Report Generation with CSV/Excel logging
* Mobile Camera Support using DroidCam
* Seat Vibration Simulation for driver alerts
* Real-Time Graphs for EAR, Blink Frequency, and Fatigue Score
* Multi-Person Detection for driver/passenger identification
* Face Recognition for driver authentication
* Cloud Logging using Firebase, AWS, or MongoDB
* GPS and Emergency Alert System with location sharing and SMS alerts
* Vehicle telemetry integration
* Infrared and low-light monitoring

---

# Recommended Final Version

Recommended feature set for a strong AI project:

* Real-Time Fatigue Detection
* Driver State Classification
* Dashboard UI
* Driver Reports
* Real-Time Graphs
* Mobile Camera Support
* Night Vision Support
* Cloud Logging
* ML-Based Driver Classification

---

# Applications

* Driver Monitoring Systems
* Advanced Driver Assistance Systems (ADAS)
* Fleet Safety Monitoring
* Automotive Safety Research
* Human Attention Monitoring
* Industrial Operator Monitoring
* Smart Transportation Systems

---

# Developed Using

* Computer Vision
* Machine Learning Concepts
* Real-Time Video Processing
* Human Behavior Analysis

---

# Future Scope

This project can be extended into:

* ADAS Systems
* Smart Transportation Platforms
* Automotive AI Systems
* Embedded Edge AI Devices
* Autonomous Vehicle Safety Platforms

---

# Author

Tushar Kumar

Mechanical Engineering
Indian Institute of Technology Roorkee

---

# License

This project is for educational and research purposes.
