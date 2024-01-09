import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

class MyAwesomeModel(pl.LightningModule):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, criterion) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # [B, 64, 24, 24] -> [B, 64, 12, 12]
            nn.Flatten(),  # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
        )
        self.fc = nn.Linear(64 * 12 * 12, 10)
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
        x = self.conv_layers(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        # self.logger.experiment is the same as wandb.log
        self.logger.experiment.log({'logits': wandb.Histogram(preds.cpu().detach().numpy())})
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# import torch
# import torch.nn as nn

# class MyAwesomeModel(torch.nn.Module):
#     """Basic neural network class.

#     Args:
#         in_features: number of input features
#         out_features: number of output features

#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2),  # [B, 64, 24, 24] -> [B, 64, 12, 12]
#             nn.Flatten(),  # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
#         )
#         self.fc = nn.Linear(64 * 12 * 12, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass of the model.

#         Args:
#             x: input tensor expected to be of shape [N,in_features]

#         Returns:
#             Output tensor with shape [N,out_features]

#         """
#         x = self.conv_layers(x)
#         return self.fc(x)

#     def extract_features(self, x: torch.Tensor) -> torch.Tensor:
#         """Extract intermediate features from the model.

#         Args:
#             x: input tensor expected to be of shape [N,in_features]

#         Returns:
#             Output tensor with shape [N, 64 * 12 * 12]

#         """
#         return self.conv_layers(x)
