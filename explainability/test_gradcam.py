import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from models.efficientnet import get_efficientnet
from explainability.gradcam import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load trained model
# -------------------------
model = get_efficientnet(num_classes=4)
model.load_state_dict(
    torch.load("checkpoints/efficientnet_best.pth", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# -------------------------
# IMPORTANT FIX: target layer
# (last CONV layer, not block)
# -------------------------
target_layer = model.features[-1][0]
gradcam = GradCAM(model, target_layer)

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

# -------------------------
# Load image
# -------------------------
IMAGE_PATH = "sample_mri.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        "Place an MRI image named 'sample_mri.jpg' in the project root."
    )

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# -------------------------
# Forward + backward
# -------------------------
model.zero_grad()

output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# Class-specific backward (REQUIRED)
output[0, pred_class].backward()

# -------------------------
# Generate Grad-CAM
# -------------------------
cam = gradcam.generate_cam()

# -------------------------
# Overlay heatmap
# -------------------------
image_np = np.array(image.resize((224, 224)))
heatmap = cv2.applyColorMap(
    np.uint8(255 * cam),
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

# -------------------------
# Save output
# -------------------------
cv2.imwrite("gradcam_output.jpg", overlay)
print("✅ Grad-CAM saved successfully as gradcam_output.jpg")
