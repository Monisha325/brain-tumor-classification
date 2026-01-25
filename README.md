# 🧠 Brain Tumor Classification from MRI using CNNs

An end-to-end deep learning application that classifies brain MRI images into tumor categories using Convolutional Neural Networks (CNNs), with explainability and live deployment.

---

## 🌍 Live Demo

- **Streamlit Web App (Frontend):**  
  https://brain-tumor-classification-qs2tddfuoe264cnrdqx3to.streamlit.app/

- **FastAPI Backend (Render):**  
  https://brain-tumor-classification-2911.onrender.com/

- **API Documentation (Swagger UI):**  
  https://brain-tumor-classification-2911.onrender.com/docs

---

## 📌 Problem Statement

Brain tumor diagnosis from MRI scans is a time-consuming and expert-driven process.  
This project aims to assist radiologists by automatically classifying brain MRI images using deep learning models.

> ⚠️ This system is intended as a **decision-support tool**, not a diagnostic system.

---

## 🧠 Tumor Classes

The model predicts one of the following classes:

- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

---

## 🚀 Models Used

### 🔹 Primary Model
- **EfficientNet-B0**
- Selected for high accuracy with fewer parameters

### 🔹 Secondary Model
- **DenseNet-121**
- Used for performance comparison

### 🔹 Training Strategy
- Transfer learning using ImageNet pretrained weights  
- Fine-tuning on MRI dataset

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM is used to visualize important regions of the MRI image that influence the model’s prediction.

This helps in:
- Improving interpretability
- Building trust in medical AI systems
- Validating model focus on tumor regions

---

## 📊 Model Evaluation

The model is evaluated using standard classification metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

> **Recall is prioritized** due to the critical nature of false negatives in medical applications.

---

## 🌐 Backend API (FastAPI)

The trained model is deployed as a REST API using FastAPI.

### Endpoint
POST /predict


### Input
- Brain MRI image (JPG / PNG)

### Output
```json
{
  "prediction": "Meningioma",
  "confidence": 0.72
}
🖥️ Frontend (Streamlit)
A Streamlit web application provides an interactive interface to:

Upload MRI images

Receive predicted tumor class

View prediction confidence

The frontend communicates with the FastAPI backend for inference.

🧱 Project Structure
brain_tumor/
├── api/                     # FastAPI backend
│   └── main.py
│
├── app/                     # Streamlit frontend
│   └── app.py
│
├── data/                    # Dataset utilities (dataset not pushed)
│   ├── dataset.py
│   ├── loader.py
│   └── transforms.py
│
├── models/                  # Model architectures
│   ├── efficientnet.py
│   └── densenet.py
│
├── training/                # Training scripts
│   ├── train.py
│   └── train_densenet.py
│
├── evaluation/              # Evaluation and metrics
│   ├── metrics.py
│   └── evaluate.py
│
├── explainability/          # Explainable AI (Grad-CAM)
│   ├── gradcam.py
│   └── test_gradcam.py
│
├── checkpoints/             # Trained model weights
│   ├── efficientnet_best.pth
│   └── densenet_best.pth
│
├── requirements.txt
└── README.md
▶️ Run Locally (Optional)
Install Dependencies
pip install -r requirements.txt
Start Backend
uvicorn api.main:app --reload
Start Frontend
streamlit run app/app.py
⚠️ Disclaimer
This project is intended only for educational and research purposes.
It must not be used for clinical or diagnostic decision-making.

👤 Author
Monisha Patnana
3rd Year Undergraduate Student
GITAM University

⭐ Key Highlights
End-to-end ML system

CNN-based medical image classification

Explainable AI using Grad-CAM

FastAPI backend deployment

Streamlit frontend deployment

Fully hosted and publicly accessible