import torch
import torch.nn as nn
from torch.optim import Adam
from data.loader import get_loaders
from models.efficientnet import get_efficientnet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    train_loader, test_loader = get_loaders(
        "data/Training",
        "data/Testing"
    )

    model = get_efficientnet(num_classes=4).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    best_acc = 0

    for epoch in range(5):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch+1} | Train Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/efficientnet_best.pth")

if __name__ == "__main__":
    train()
