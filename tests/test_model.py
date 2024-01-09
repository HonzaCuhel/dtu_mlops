import pytest
import torch
import torch.nn as nn
from dtu_mlops_code.models.model import MyAwesomeModel


def test_model():
    loss_fn = nn.CrossEntropyLoss()
    model = MyAwesomeModel(loss_fn)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)

# tests/test_model.py
def test_error_on_wrong_shape():
    loss_fn = nn.CrossEntropyLoss()
    model = MyAwesomeModel(loss_fn)
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
