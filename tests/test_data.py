from dtu_mlops_code.data.make_dataset import mnist
import pytest

def check_dataset(dataset):
    for x, y in dataset:
        assert x.shape == (1, 28, 28) or x.shape == (784,)
        assert y in range(10)


def test_data():
    train_dataset, test_dataset = mnist()
    print(train_dataset)
    assert len(train_dataset) == 25000 and len(test_dataset) == 5000, "Dataset did not have the correct number of samples"
    # assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    # assert that all labels are represented
    check_dataset(train_dataset)
    check_dataset(test_dataset)
    # for x, y in train_dataset:
    #     assert x.shape == (1, 28, 28)
    #     assert y in range(10)
    # for x, y in test_dataset:
    #     assert x.shape == (1, 28, 28)
    #     assert y in range(10)

# coverage run -m pytest tests/
# coverage report
# coverage report -m