import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, image_file, label_file):
        self.images = torch.load(image_file).to(torch.float32)
        self.labels = torch.load(label_file) # torch.nn.functional.one_hot(torch.load(label_file), num_classes=10)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].view((-1))
        label = self.labels[idx]
        return image, label


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)
    # train = torch.load("../../../data/corruptmnist/train_images_0.pt")
    # test = torch.randn(10000, 784)
    # return train, test
    training_data = CustomDataset(
        "../../../data/corruptmnist/train_images_0.pt", 
        "../../../data/corruptmnist/train_target_0.pt"
    )
    test_dataloader = CustomDataset(
        "../../../data/corruptmnist/test_images.pt", 
        "../../../data/corruptmnist/test_target.pt"
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataloader, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader
