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
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@hydra.main(config_path="./conf", config_name="training_conf.yaml")
def train(cfg):
    """Train a model on MNIST."""
    log.info("Training day and night")
    log.info(str(cfg.hyperparameters.lr))
    log.info(str(cfg.hyperparameters.batch_size))

    # TODO: Implement training loop here
    loss_fn = nn.CrossEntropyLoss()
    model = MyAwesomeModel(loss_fn).to(device)
    train_set, test_set = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    # losses = []  # Initialize the losses list
    checkpoint_file = f"{os.getcwd()}/trained_model.pt"

    trainer = Trainer(logger=pl.loggers.WandbLogger(project="dtu_mlops"), max_epochs=cfg.hyperparameters.num_epochs)
    # trainer = Trainer(max_epochs=cfg.hyperparameters.num_epochs)
    trainer.fit(model, train_dataloader)

    # torch.save(model, "./models/trained_model.pt")
    torch.save(model, checkpoint_file)
    trainer.validate(model, test_dataloader)
    # evaluate(checkpoint_file)

    preds, target = [], []
    for batch in train_dataloader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open(to_absolute_path("classification_report.txt"), 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
    plt.savefig(to_absolute_path('confusion_matrix.png'))


if __name__ == "__main__":
    # cli()
    train()
