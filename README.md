🧠 MindScreen – Multimodal Depression Detection System

An AI-powered system that analyzes facial expressions and vocal patterns from video input to detect depressive indicators and generate a structured mental health report.

---

🎯 Overview

MindScreen is a multimodal deep learning pipeline that integrates computer vision and audio processing to assess behavioral signals associated with depression.

Users can:

- 🎥 Upload a video
- 🧠 Analyze facial + audio features
- 📄 Generate an automated PDF report

---

⚙️ System Pipeline

Video Input → Feature Extraction → Multimodal Fusion Model → Report Generation

🔹 Components

- Facial Analysis: Extracts facial behavior features using OpenFace
- Audio Processing: Captures vocal cues using Librosa
- Fusion Model: Combines modalities using deep learning
- Report Generator: Produces structured PDF output

---

🧠 Tech Stack

- Deep Learning: PyTorch
- Computer Vision: OpenCV, OpenFace
- Audio Processing: Librosa
- Backend: FastAPI
- Frontend: Streamlit
- Report Generation: ReportLab

---

🚀 Features

- ✅ Multimodal analysis (audio + video)
- ✅ Real-time video processing
- ✅ Automated PDF report generation
- ✅ End-to-end AI pipeline
- ✅ API-based architecture

---

📊 Model Approach

- Feature extraction from:
  
  - Facial landmarks & expressions
  - Audio signals

- Fusion strategy:
  
  - Combined feature representation
  - Deep learning-based classification

---

⚠️ Limitations

- Requires good video/audio quality
- Model trained on limited datasets
- Not a medical diagnosis tool
- Performance may vary in real-world conditions

---

📦 Installation
```bash
git clone https://github.com/shahir55312-collab/DeepLearningDepressionDetection
cd DeepLearningDepressionDetection
pip install -r requirements.txt
```

▶️ Running the Project

🔹 Run Backend (FastAPI)

uvicorn backend.main:app --reload

🔹 Run Frontend (Streamlit)

streamlit run frontend/app.py

---

🌐 Live Demo

⚠️ Due to heavy dependencies (OpenFace, model size), full deployment is not hosted permanently.

👉 Live demo available on request
👉 Temporary demo available via ngrok

---

🧪 Future Improvements

- 📈 Improve model accuracy
- 🔍 Add explainable AI (feature importance)
- 📊 Enable long-term behavioral tracking
- ☁️ Deploy scalable cloud infrastructure

---

🤝 Acknowledgment

- Open-source libraries and tools
- AI-assisted development tools (ChatGPT, etc.)

---

📌 Disclaimer

This project is intended for research and educational purposes only and should not be used as a substitute for professional medical diagnosis.

---

👨‍💻 Author

Shahir Ali
First Year Engineering Student

---

⭐ If you like this project

Give it a ⭐ on GitHub!
