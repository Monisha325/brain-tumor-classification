from data.loader import get_loaders

train_loader, test_loader = get_loaders(
    "data/Training",
    "data/Testing"
)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)
