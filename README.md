# Smart Fall Detection System
AI-Powered Real-Time Fall Detection using Deep Learning and Computer Vision.

---

## Overview

The Smart Fall Detection System is a real-time AI-based safety monitoring application that detects human falls using a fine-tuned MobileNetV2 deep learning model.

This system is optimized for CPU performance and provides a professional GUI interface for live monitoring. It includes an intelligent multi-frame confirmation mechanism to reduce false alarms and improve detection accuracy.

---

## Features

- Real-time webcam monitoring
- Fall / Not-Fall classification
- Transfer learning using MobileNetV2
- Multi-frame fall confirmation logic
- Alarm system with automatic stop
- Screenshot capture on fall detection
- Alert counter system
- CPU-optimized inference
- Professional Tkinter GUI

---

## Model Architecture

- Pretrained MobileNetV2 (Transfer Learning)
- Custom binary classification head
- Data augmentation for improved generalization
- Multi-frame temporal smoothing for improved accuracy

---

## Performance

- Training Accuracy: Approximately 88–90%
- Optimized for CPU-only systems
- Frame-skipping for faster real-time detection
- Reduced false positives using multi-frame validation

---

## System Workflow

1. Capture live video from webcam
2. Preprocess frame (resize and normalize)
3. Perform inference using trained MobileNetV2 model
4. Confirm fall detection across multiple frames
5. Trigger alert system:
   - Play alarm
   - Save screenshot
   - Increase alert counter
6. Stop alarm immediately when fall condition clears

---

## Project Structure

Smart-Fall-Detection-System/
│
├── gui_app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── alerts/

---

## Technologies Used

- Python
- PyTorch
- MobileNetV2
- OpenCV
- Tkinter
- Pygame
- Matplotlib
- Seaborn
- Scikit-learn

---

## Installation

### Clone Repository

git clone https://github.com/yourusername/Smart-Fall-Detection-System.git
cd Smart-Fall-Detection-System

### Install Dependencies

pip install -r requirements.txt

### Run Application

python gui_app.py

---

## Training the Model

python train.py

---

## Model Evaluation

python evaluate.py

---

## Use Cases

- Smart campus surveillance
- Elderly monitoring systems
- AI-based safety applications
- Intelligent CCTV monitoring
- Healthcare monitoring systems

---

## Author

Mohamed Mustak M  
AI and Full Stack Developer  
Specialized in Computer Vision and Intelligent Systems
