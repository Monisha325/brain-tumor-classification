# 🧠 Brain Tumor Classification from MRI using CNNs

An end-to-end deep learning system to classify brain MRI images into tumor categories using Convolutional Neural Networks (CNNs), with a deployed backend API and an interactive frontend.

This project is designed as a **decision-support system**, not a diagnostic tool.

---

## 🌍 Live Deployment

- **Streamlit Web App (Frontend):**  
  https://brain-tumor-classification-qs2tddfuoe264cnrdqx3to.streamlit.app/

- **FastAPI Backend (Render):**  
  https://brain-tumor-classification-2911.onrender.com/

- **API Documentation (Swagger UI):**  
  https://brain-tumor-classification-2911.onrender.com/docs

---

## 📌 Problem Statement

Manual analysis of brain MRI scans is time-consuming and requires expert radiologists.  
The objective of this project is to build an automated system that can assist in **classifying brain tumors from MRI images** using deep learning techniques.

---

## 🧠 Tumor Classes

The model classifies MRI images into the following four categories:

- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

---

## 🚀 Models Used

### Primary Model
- **EfficientNet-B0**
- Chosen for its strong performance and parameter efficiency

### Secondary Model
- **DenseNet-121**
- Used for architectural comparison

### Training Strategy
- Transfer learning with ImageNet pretrained weights  
- Fine-tuning final layers for medical image adaptation

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the regions of MRI images that influence the model’s predictions.

This improves:
- Interpretability of predictions
- Trust in medical AI systems
- Validation of model focus on tumor-relevant regions

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

**Recall is emphasized**, as false negatives are critical in healthcare-related applications.

---

## 🌐 Backend API (FastAPI)

The trained model is exposed through a FastAPI-based inference service.

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

A Streamlit web application allows users to:

Upload MRI images

Receive predicted tumor class

View confidence scores

The frontend communicates with the FastAPI backend for inference.

🏗️ Project Structure

The project follows a clean, modular architecture:

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
Install dependencies
pip install -r requirements.txt

Run backend
uvicorn api.main:app --reload

Run frontend
streamlit run app/app.py

⚠️ Disclaimer

This project is intended for educational and research purposes only.
It is not a medical diagnostic system and should not be used for clinical decision-making.

👤 Author

Monisha Patnana
3rd Year Undergraduate Student
GITAM University

This project was developed as a 3rd year academic and portfolio project, focusing on:

Deep Learning

Explainable AI

Medical Image Analysis

API Development

End-to-End ML Deployment


---

## ✅ Final steps (last time)

```bash
git add README.md
git commit -m "Add final README with live deployment links"
git push