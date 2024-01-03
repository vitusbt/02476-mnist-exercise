import click
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
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

    model = MyAwesomeModel()
    train_set, _ = mnist()

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    epochs = 50

    for ep in range(epochs):
        running_loss = 0
        for step, (images, labels) in enumerate(train_set):
            optimizer.zero_grad()
            labels_pred = model(images)
            loss = loss_fn(labels_pred, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print(f"Epoch {ep+1}/{epochs} | Training loss: {running_loss/len(train_set)}")
    else:
        torch.save(model, 'model_mnist.pt')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    model.eval()
    _, test_set = mnist()

    with torch.no_grad():
        correct = 0
        total = 0
        for step, (images, labels) in enumerate(test_set):
            labels_pred = torch.argmax(model(images), dim=1)
            correct += torch.sum(labels_pred == labels).item()
            total += labels.shape[0]

        acc = correct/total
        print(f'Correct: {correct}')
        print(f'Total: {total}')
        print(f'Accuracy: {acc*100}%')



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
