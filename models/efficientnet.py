import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes):
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)

    # -----------------------------
    # Freeze all feature layers
    # -----------------------------
    for param in model.features.parameters():
        param.requires_grad = False

    # 🔥 IMPORTANT FOR GRAD-CAM
    # Unfreeze the LAST convolution block only
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # -----------------------------
    # Replace classifier
    # -----------------------------
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model
