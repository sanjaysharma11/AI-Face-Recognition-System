# 🎯 Face Recognition System using Python, OpenCV & Deep Learning

This is a simple AI/ML-based **Face Recognition System** built with Python 3.9.5, OpenCV, and the `face_recognition` library (built on top of dlib’s deep learning model). It identifies and labels faces in test images based on trained face encodings.

## 🧠 Tech Stack

- **Language**: Python 3.9.5  
- **Libraries**:
  - `face_recognition`
  - `dlib==19.22.1`
  - `OpenCV` (`cv2`)
  - `NumPy==1.26.4`
  - `os` (standard library)

---

## 🚀 Features

- Detects and encodes faces from training images.
- Compares faces from test images with known encodings.
- Labels and displays recognized faces using bounding boxes.
- Logs unmatched or missing faces with appropriate messages.

---


> 📸 **Note**: File names in `train/` must be in the format `name.jpg`, `name.png`, etc. The filename (before extension) will be used as the person's label.

---

## 📦 Installation Guide

### 1. 🐍 Python Version

Ensure you're using **Python 3.9.5**.

```bash
python --version
# Output should be: Python 3.9.5

pip install dlib-19.22.1-cp39-cp39-win_amd64.whl
pip install face_recognition
pip install numpy==1.26.4  
