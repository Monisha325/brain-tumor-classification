from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import os

from models.efficientnet import get_efficientnet

app = FastAPI(title="Brain Tumor Classification API")

# -------------------------
# Load model (CPU only)
# -------------------------
DEVICE = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "efficientnet_best.pth")

model = get_efficientnet(num_classes=4)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# -------------------------
# Health check
# -------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        return JSONResponse({
            "prediction": CLASSES[pred.item()],
            "confidence": round(confidence.item(), 4)
        })

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
