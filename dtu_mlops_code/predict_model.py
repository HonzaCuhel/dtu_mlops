import torch
import numpy as np
from models import MyAwesomeModel


def predict(model_path: str, data_path: str) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    # Load the model
    model = MyAwesomeModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load the data
    data = np.load(data_path)
    data = torch.from_numpy(data).float()

    # Generate predictions
    with torch.no_grad():
        predictions = model(data)

    return predictions
