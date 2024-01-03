import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(f"./data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"./data/raw/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("./data/raw/test_images.pt")
    test_labels = torch.load("./data/raw/test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # Normalize train_data and test_data
    train_data = (train_data - train_data.mean()) / train_data.std()
    test_data = (test_data - test_data.mean()) / test_data.std()

    torch.save(train_data, "./data/processed/train_images.pt")
    torch.save(train_labels, "./data/processed/train_labels.pt")
    torch.save(test_data, "./data/processed/test_images.pt")
    torch.save(test_labels, "./data/processed/test_labels.pt")

    return (
        torch.utils.data.TensorDataset(train_data, train_labels),
        torch.utils.data.TensorDataset(test_data, test_labels),
    )


if __name__ == "__main__":
    # Get the data and process it
    mnist()
