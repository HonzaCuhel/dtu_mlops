import click
import torch
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainloader, testloader = mnist()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()
    running_loss = 0
    epochs = 30
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()

            output = model.forward(images)
            # print(images.shape, output.shape, labels.shape, images.dtype)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Model in inference mode, dropout is off
        model.eval()

        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            accuracy = 0
            test_loss = 0
            for images, labels in testloader:
                output = model.forward(images)
                test_loss += criterion(output, labels).item()

                ## Calculating the accuracy
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output)
                # Class with highest probability is our predicted class, compare with true label
                equality = labels.data == ps.max(1)[1]
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

        print(
            "Epoch: {}/{}.. ".format(e + 1, epochs),
            "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
        )

        running_loss = 0
    torch.save(model.state_dict(), 'trained_model.pt')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    _, test_set = mnist()

    # Model in inference mode, dropout is off
    model.eval()

    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        accuracy = 0
        for images, labels in test_set:
            output = model.forward(images)
            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = labels.data == ps.max(1)[1]
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Accuracy: {:.3f}".format(accuracy / len(test_set)))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

# python main.py train --lr 1e-4
# python main.py evaluate trained_model.pt