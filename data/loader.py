from torch.utils.data import DataLoader
from data.dataset import BrainMRIDataset
from data.transforms import train_transforms, test_transforms

def get_loaders(train_path, test_path, batch_size=32):
    train_dataset = BrainMRIDataset(train_path, train_transforms)
    test_dataset = BrainMRIDataset(test_path, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
