<div align="center">

# 🔍 DeepDetect V2

### AI-Powered Deepfake Detection Made Simple

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

**Can you trust what you see online? DeepDetect V2 helps you find out.**

 • [Getting Started](#-quick-start) • [How It Works](#-how-it-works) • [Performance](#-performance-metrics)


---

</div>

## 🎯 What is DeepDetect V2?

In an era where AI-generated content is becoming indistinguishable from reality, **DeepDetect V2** empowers anyone to verify the authenticity of images and videos. Simply upload your media, and our advanced AI analyzes it in seconds—no technical knowledge required.

### ✨ Why Choose DeepDetect V2?

- **🚀 Lightning Fast** - Get results in seconds, not minutes
- **🎓 Beginner Friendly** - No AI expertise needed
- **🤖 AI-Explained Results** - Understand *why* something was flagged as fake
- **🎯 High Accuracy** - 97% accuracy rate on benchmark datasets
- **🌐 Web-Based** - Access from anywhere, no installation needed

---

## 🎬 See It In Action

<div align="center">

### Upload → Analyze → Understand

```mermaid
graph LR
    A[📤 Upload Image/Video] --> B[👤 Detect Faces]
    B --> C[🔬 Deep Learning Analysis]
    C --> D[🎯 Real or Fake?]
    D --> E[💬 AI Explanation]
    
    style A fill:#4A90E2,color:#fff
    style C fill:#E24A4A,color:#fff
    style E fill:#4AE290,color:#fff
```

<table>
<tr>
<td width="33%">

<p><b>Upload Your Media</b><br/>Images or videos in seconds</p>
</td>
<td width="33%">
<p><b>AI Analysis</b><br/>MesoNet scans for manipulation</p>
</td>
<td width="33%">

<p><b>Clear Results</b><br/>Easy-to-understand verdict</p>
</td>
</tr>
</table>

</div>

---

## 🛠️ How It Works

### The Technology Behind The Magic

DeepDetect V2 combines cutting-edge AI technologies to provide reliable deepfake detection:

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Upload  →  Face Detection  →  MesoNet Model  →  Gemini AI │
│   (Image/Video)     (MTCNN)      (Deep Learning)   (Explanation) │
└─────────────────────────────────────────────────────────────────┘
```

#### 🧠 The Detection Process

1. **Face Extraction** - MTCNN technology locates and extracts the primary face from your media
2. **Microscopic Analysis** - MesoNet deep learning model examines pixel-level inconsistencies invisible to the human eye
3. **Confidence Scoring** - The model assigns a confidence score (0-100%) indicating likelihood of manipulation
4. **AI Explanation** - Google's Gemini AI analyzes the visual evidence and explains the verdict in plain English

#### 💻 Tech Stack

<div align="center">

| Category | Technology |
|----------|-----------|
| **Backend** | Python, Flask |
| **AI/ML** | TensorFlow, Keras, MesoNet Architecture |
| **Face Detection** | MTCNN (Multi-task Cascaded CNN) |
| **AI Explanations** | Google Gemini Pro Vision API |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Render, Gunicorn |

</div>

---

## 📊 Performance Metrics

### Real-World Accuracy You Can Trust

Tested on the industry-standard **FaceForensics++** benchmark dataset with over 1,000 videos.

<div align="center">

| Metric | Score | What It Means |
|--------|-------|---------------|
| **🎯 Accuracy** | 97% | Overall correctness across all predictions |
| **✅ Precision** | 98% | When flagged as FAKE, it's actually fake 98% of the time |
| **🔍 Recall** | 96% | Catches 96% of all actual deepfakes |
| **⚖️ F1-Score** | 97% | Balanced performance measure |

</div>

### 📈 Visual Performance Analysis

<div align="center">

#### Confusion Matrix - Prediction Breakdown

```
                  Predicted
                REAL    FAKE
Actual REAL     485      12      ← 97.6% correct
       FAKE      19     484      ← 96.2% caught
```


*Strong diagonal shows excellent classification performance*

---

#### ROC Curve - Detection Capability

**AUC Score: 0.98** - Near-perfect ability to distinguish real from fake

</div>

### ⚡ Speed Performance

- **Images**: < 3 seconds average processing time
- **Videos**: ~1 second per second of video footage
- **API Response**: < 2 seconds for Gemini explanations

---

## 🚀 Quick Start

### Run Locally in 5 Minutes

**Prerequisites**: Python 3.8+ installed on your system

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/deepdetect-v2.git
cd deepdetect-v2

# 2️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Set up your Gemini API key
export GEMINI_API_KEY="your_api_key_here"
# Windows CMD: set GEMINI_API_KEY=your_api_key_here
# Windows PowerShell: $env:GEMINI_API_KEY="your_api_key_here"

# 5️⃣ Ensure model file is in place
# Place your .hdf5 model in the model/ directory

# 6️⃣ Launch the app
python app.py
```

Open your browser to `http://127.0.0.1:8080` and start detecting! 🎉

### 🔑 Getting Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy and use in the setup above

---




## 🔮 Roadmap & Future Enhancements

- [ ] **Multi-Model Support** - Let users choose from different detection algorithms
- [ ] **Batch Processing** - Analyze multiple files simultaneously
- [ ] **Enhanced UI/UX** - More intuitive design with real-time progress
- [ ] **Performance Dashboard** - Interactive metrics visualization
- [ ] **Model Optimization** - ONNX conversion for faster inference
- [ ] **Mobile App** - Native iOS and Android applications
- [ ] **API Access** - RESTful API for developers

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 About the Developer

<div align="center">

**Built with ❤️ by [Your Name]**

[🌐 Portfolio](https://vendotha.onrender.com) • [💼 LinkedIn](https://linkedin.com/in/vendotha) • [📧 Email](mailto:vendotha@gmail.com)

*Passionate about AI, computer vision, and building tools that make technology accessible to everyone.*

---

### 🌟 If you find this project useful, please consider giving it a star!

[![GitHub stars](https://img.shields.io/github/stars/vendotha/deepdetect-v2?style=social)](https://github.com/vendotha/deepdetect-v2)

</div>

---

<div align="center">

**🛡️ Fighting Misinformation, One Image at a Time**

Made with Python 🐍 • TensorFlow 🧠 • Gemini AI ✨

</div>
