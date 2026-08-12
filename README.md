# 🔍 DeepDetect V2

### Spatial Frequency Deepfake Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

**DeepDetect V2 is a deepfake detection system that combines spatial visual features with frequency-domain signals to identify manipulated media more effectively.**

• Getting Started • Architecture • Performance • Research

---

## 🎯 Overview

Deepfake generation techniques continue to improve and often bypass conventional visual inspection methods. DeepDetect V2 addresses this by combining:

* **Spatial image features (RGB domain)**
* **Frequency-domain signals using DCT**
* **Deep learning inference pipelines**
* **Image and video analysis workflows**

The system is designed to improve robustness against compressed media and subtle manipulations often missed by spatial-only approaches.

---

## ✨ Features

* 🚀 Fast image and video inference
* 🎯 Dual-stream spatial + frequency analysis
* 📊 Confidence-based prediction scores
* 🔍 Frequency artifact detection
* 🌐 Browser-based interface
* 🐳 Dockerized deployment support

---

## 🧠 System Architecture

```text
Image / Video
      ↓
Face Detection (MTCNN)
      ↓
Face Extraction Pipeline
      ↓

 ┌─────────────────────────┐
 │ RGB Spatial Features    │
 └─────────────────────────┘

 ┌─────────────────────────┐
 │ DCT Frequency Features  │
 └─────────────────────────┘

            ↓
Feature Fusion Layer
            ↓
Dual-Stream CNN
            ↓
Prediction Layer
            ↓
Real / Deepfake
```

---

## 🛠 Detection Pipeline

### Step 1  Face Extraction

Media is processed using MTCNN to detect and isolate facial regions.

### Step 2  Spatial Feature Processing

The RGB branch learns visual inconsistencies and manipulation artifacts.

### Step 3  Frequency Feature Processing

DCT transforms are applied to identify hidden frequency-domain patterns often introduced during synthetic generation.

### Step 4  Feature Fusion

Spatial and frequency representations are combined before classification.

### Step 5  Prediction

The model outputs confidence scores for real and manipulated media.

---

## 💻 Tech Stack

| Category           | Technology            |
| ------------------ | --------------------- |
| Backend            | Python, Flask         |
| Deep Learning      | TensorFlow, Keras     |
| Face Detection     | MTCNN                 |
| Frequency Analysis | DCT                   |
| Deployment         | Docker, Gunicorn      |
| Frontend           | HTML, CSS, JavaScript |

---

## 📊 Performance

Evaluated on benchmark datasets containing image and video deepfake samples.

| Metric                     |                  Score |
| -------------------------- | ---------------------: |
| Accuracy                   |                    92% |
| Precision                  |                    90% |
| Frequency Branch Detection |                    85% |
| Dataset Size               | 10k images + 1k videos |

---

## 📈 Comparative Performance

| Feature                     | Spatial CNN | DeepDetect V2 |
| --------------------------- | ----------: | ------------: |
| RGB Features                |           ✓ |             ✓ |
| Frequency Features          |           ✗ |             ✓ |
| Robustness to Compression   |    Moderate |          High |
| Detects Frequency Artifacts |           ✗ |             ✓ |

---

## 🚀 Quick Start

```bash
git clone https://github.com/vendotha/DeepDetect-V2.git

cd DeepDetect-V2

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python app.py
```

Open:

http://127.0.0.1:8080

---

## 🔬 Research Context

This project served as the implementation foundation for:

**Spectra-FakeNet: Spatial-Frequency Deepfake Detection**

**Paper Link:** [Spectra FakeNet →](https://github.com/vendotha/Spectra-FakeNet)


The work explores combining RGB spatial information and DCT frequency-domain features for more robust deepfake detection.

---

## 👨‍💻 Developer

Buvananand Vendotha

Portfolio: https://vendotha.web.app

LinkedIn: https://linkedin.com/in/vendotha
