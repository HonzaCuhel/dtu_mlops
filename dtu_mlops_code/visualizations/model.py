import torch
import torch.nn as nn


class MyAwesomeModel(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = self.conv_layers(x)
        return self.fc(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N, 64 * 12 * 12]

        """
        return self.conv_layers(x)
