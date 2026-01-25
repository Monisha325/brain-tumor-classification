import torch
from tqdm import tqdm

from data.loader import get_loaders
from models.efficientnet import get_efficientnet
from evaluation.metrics import compute_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    _, test_loader = get_loaders(
        "data/Training",
        "data/Testing",
        batch_size=32
    )

    model = get_efficientnet(num_classes=4)
    model.load_state_dict(
        torch.load("checkpoints/efficientnet_best.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)

    print("\n📊 Evaluation Results")
    print("---------------------")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])

if __name__ == "__main__":
    evaluate()
