# 🧠 Brain Tumor Classification from MRI using CNNs

An end-to-end deep learning system that **identifies and classifies brain tumors from MRI images** using CNN-based transfer learning, with **explainable AI (Grad-CAM)** and **full-stack deployment**.

> 🔍 Built as a **decision-support system** using medical imaging, deep learning, and modern ML deployment practices.

---

## ⭐ Key Highlights

- CNN-based brain tumor classification from MRI images  
- Transfer learning with **EfficientNet-B0** and **DenseNet-121**  
- Explainable AI using **Grad-CAM** for visual interpretation  
- REST API deployed using **FastAPI**  
- Interactive frontend deployed using **Streamlit**  
- Fully hosted and publicly accessible system  

---

## 📁 Dataset

The dataset is **not included in this repository** due to size constraints.

You can download the Brain MRI dataset from Kaggle:

🔗 **Kaggle Dataset Link:**  
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

### Dataset Classes
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

### Dataset Structure (after download)
```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```
---

## 🌍 Live Deployment

- **Streamlit Web App (Frontend):**  
  https://brain-tumor-classification-qs2tddfuoe264cnrdqx3to.streamlit.app/

- **FastAPI Backend (Render):**  
  https://brain-tumor-classification-2911.onrender.com/

- **API Documentation (Swagger UI):**  
  https://brain-tumor-classification-2911.onrender.com/docs

---

## 🚀 Project Overview

Manual analysis of brain MRI scans is time-consuming and requires expert radiologists.  
This project aims to assist medical professionals by **automatically classifying brain MRI images** into tumor categories using Convolutional Neural Networks (CNNs).

The system focuses on **accuracy, interpretability, and deployability**.

---

## 🧠 Solution Approach

1. Dataset preparation and preprocessing  
2. Data augmentation for robustness  
3. CNN-based model training using transfer learning  
4. Model evaluation using medical classification metrics  
5. Explainable AI using Grad-CAM  
6. Backend deployment using FastAPI  
7. Interactive web interface using Streamlit  

---

## 🛠 Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Scikit-learn**
- **OpenCV**
- **FastAPI**
- **Streamlit**

---

## 🧠 Tumor Classes

The model classifies MRI images into the following categories:

- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

---

## 📂 Project Structure

```text
brain_tumor/
│
├── api/                       # FastAPI backend
│   └── main.py
│
├── app/                       # Streamlit frontend
│   └── app.py
│
├── data/                      # Dataset utilities (dataset not pushed)
│   ├── dataset.py
│   ├── loader.py
│   └── transforms.py
│
├── models/                    # Model architectures
│   ├── efficientnet.py
│   └── densenet.py
│
├── training/                  # Training scripts
│   ├── train.py
│   └── train_densenet.py
│
├── evaluation/                # Evaluation and metrics
│   ├── metrics.py
│   └── evaluate.py
│
├── explainability/            # Grad-CAM implementation
│   ├── gradcam.py
│   └── test_gradcam.py
│
├── checkpoints/               # Trained model weights
│   ├── efficientnet_best.pth
│   └── densenet_best.pth
│
├── requirements.txt
└── README.md
```
---

## 📊 Model Performance
The models were evaluated using standard medical image classification metrics.

### Metrics Used
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📊 Model Evaluation Comparison

### 🔹 Overall Performance

| Metric    | EfficientNet-B0 | DenseNet-121 |
|------------|-----------------|--------------|
| Accuracy   | 0.8733 (87%)    | 0.8649 (86%) |
| Precision  | 0.8786 (88%)    | 0.8658 (86%) |
| Recall     | 0.8733 (87%)    | 0.8649 (86%) |
| F1-score   | 0.8712 (87%)    | 0.8624 (86%) |

### 🔹 Confusion Matrix (EfficientNet-B0)
[[234 55 3 7]
[ 2 221 71 12]
[ 1 1 402 1]
[ 0 13 0 287]]


### 🔹 Confusion Matrix (DenseNet-121)
[[241 42 1 15]
[ 11 216 41 38]
[ 4 8 387 6]
[ 2 9 0 289]]


### 🔹 Weighted Classification Summary

| Model            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| EfficientNet-B0  | 0.88      | 0.87   | 0.87     |
| DenseNet-121     | 0.87      | 0.86   | 0.86     |

## 🏆 Final Model Selection

EfficientNet-B0 was selected as the final deployment model because it consistently achieved higher accuracy, precision, recall, and F1-score compared to DenseNet-121.

Since recall is critical in medical diagnosis to minimize false negatives, EfficientNet’s higher recall and balanced performance

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM visualizes the regions of MRI images that most influence the model’s predictions, helping validate that the model focuses on tumor-relevant areas.

---

## 🌐 API Details (FastAPI)

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
```
---

## ▶️ Running the Application Locally
### 1️⃣ Install Dependencies
pip install -r requirements.txt
### 2️⃣ Start Backend API
uvicorn api.main:app --reload
### 3️⃣ Start Streamlit App
streamlit run app/app.py

---

## ⚠️ Disclaimer
This project is intended for educational and research purposes only.
It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## 👤 Author

**Monisha Patnana**  
3rd Year Undergraduate Student  
GITAM University

