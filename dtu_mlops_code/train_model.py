# import click
import torch
from torch import nn
from models import MyAwesomeModel
import matplotlib.pyplot as plt
import hydra
import logging
log = logging.getLogger(__name__)
from hydra.utils import to_absolute_path
from data import mnist
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @click.group()
# def cli():
#     """Command line interface."""
#     pass


# @click.command()
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
# @click.option("--batch_size", default=256, help="batch size to use for training")
# @click.option("--num_epochs", default=10, help="number of epochs to train for")
# def train(lr, batch_size, num_epochs):
@hydra.main(config_path="./conf", config_name="training_conf.yaml")
def train(cfg):
    """Train a model on MNIST."""
    log.info("Training day and night")
    log.info(str(cfg.hyperparameters.lr))
    log.info(str(cfg.hyperparameters.batch_size))

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)
    train_set, _ = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []  # Initialize the losses list

    for epoch in range(cfg.hyperparameters.num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch} Loss {loss}")
        losses.append(loss.item())  # Append the loss of this epoch to the losses list

    checkpoint_file = f"{os.getcwd()}/trained_model.pt"
    # torch.save(model, "./models/trained_model.pt")
    torch.save(model, checkpoint_file)
    # Plot the losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")

    evaluate(checkpoint_file)


# @click.command()
# @click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    log.info("Evaluating like my life dependends on it")
    log.info(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    log.info(str((test_preds == test_labels).float().mean()))


# cli.add_command(train)
# cli.add_command(evaluate)


if __name__ == "__main__":
    # cli()
    train()
