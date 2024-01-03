from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        
        # Dropout module with 0.4 drop probability
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        # print(self.fc1.weight.dtype, x.dtype)
        # print(x.shape)
        # Now with dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.dropout(nn.functional.relu(self.fc2(x)))
        x = self.dropout(nn.functional.relu(self.fc3(x)))
        
        # output so no dropout here
        return nn.functional.log_softmax(self.fc4(x), dim=1)
