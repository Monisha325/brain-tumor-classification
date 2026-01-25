import torch
import torch.nn as nn
from torch.optim import Adam

from data.loader import get_loaders
from models.densenet import get_densenet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    train_loader, _ = get_loaders(
        "data/Training",
        "data/Testing",
        batch_size=32
    )

    model = get_densenet(num_classes=4).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    best_acc = 0.0

    for epoch in range(5):
        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} | DenseNet Train Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                "checkpoints/densenet_best.pth"
            )

if __name__ == "__main__":
    train()
