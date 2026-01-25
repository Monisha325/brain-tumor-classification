import torch.nn as nn
from torchvision import models

def get_densenet(num_classes):
    model = models.densenet121(pretrained=True)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    return model
