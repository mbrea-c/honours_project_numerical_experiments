import os
from typing import Iterable, List, Tuple
from torch import nn
import torch
from datetime import datetime
import numpy as np
import logging
from torch.optim.optimizer import Optimizer
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class FashionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FashionNN, self).__init__()
        self.activation = nn.Softplus
        self.activation_params = {"beta": 10}
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_size, 500),
                self.activation(**self.activation_params),
            ),
            nn.Sequential(
                nn.Linear(500, 150),
                self.activation(**self.activation_params),
            ),
            nn.Sequential(
                nn.Linear(150, output_size),
            ),
        )
        # No softmax activation, automatically added by cross-entropy loss

    def forward(self, x):
        return self.layers(x)


def get_dataloaders(train_set, test_set):
    logging.info(f"Creating dataloaders")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

    return train_loader, test_loader


def get_accuracy(dataset, model):
    total = 0
    correct = 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=100)

    for images, labels in loader:
        test = images.view(-1, 28 * 28)

        outputs = model(test)

        predictions = torch.max(outputs, 1)[1]
        correct += (predictions == labels).sum()

        total += len(labels)

    accuracy = correct / total
    return accuracy


def train(
    num_epochs,
    train_loaders,
    test_loader,
    model,
    error,
    optimizers,
):
    model.train()
    count = 0
    # Lists for visualization of loss and accuracy
    train_loss_list = []
    iteration_list = []
    test_accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    if isinstance(train_loaders, DataLoader):
        assert isinstance(optimizers, Optimizer)
        train_loaders = [train_loaders]
        optimizers = [optimizers]
    else:
        assert isinstance(optimizers, List)
        assert len(optimizers) == len(train_loaders)

    n_per_epoch = sum([len(ldr) for ldr in train_loaders])

    logging.info("Training started...")
    for epoch in range(num_epochs):
        count_per_epoch = 0
        for loader, optimizer in zip(train_loaders, optimizers):
            for images, labels in loader:
                train, labels = images.view(-1, 28 * 28), labels

                # Forward pass
                outputs = model(train)
                loss = error(outputs, labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

                count += 1
                count_per_epoch += 1

                if count % 50 == 0:  # It's same as "if count % 50 == 0"
                    total = 0
                    correct = 0

                    for images, labels in test_loader:
                        labels_list.append(labels)

                        test = images.view(-1, 28 * 28)

                        outputs = model(test)

                        predictions = torch.max(outputs, 1)[1]
                        predictions_list.append(predictions)
                        correct += (predictions == labels).sum()

                        total += len(labels)

                    accuracy = correct * 100 / total
                    train_loss_list.append(loss.data)
                    iteration_list.append(count)
                    test_accuracy_list.append(accuracy)

                if count % 100 == 0:
                    logging.info(
                        f"Epoch: {epoch}/{num_epochs}, Iteration: {count_per_epoch}/{n_per_epoch}, Loss: {loss.data}, Accuracy: {accuracy}%"
                    )
    model.eval()
    return test_accuracy_list, train_loss_list, iteration_list


def train_new_model(
    train_set,
    test_set,
    learning_rate=1e-2,
    weight_decay=1e-5,
    num_epochs=5,
):
    train_loader, test_loader = get_dataloaders(train_set, test_set)

    model = FashionNN(784, 10)

    error = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    test_accuracy_list, train_loss_list, iteration_list = train(
        num_epochs=num_epochs,
        train_loaders=train_loader,
        test_loader=test_loader,
        model=model,
        error=error,
        optimizers=optimizer,
    )

    return model, test_accuracy_list, train_loss_list, iteration_list


def train_more_epochs(
    train_set,
    test_set,
    model,
    learning_rate=1e-2,
    weight_decay=1e-5,
    num_epochs=5,
):

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

    error = nn.CrossEntropyLoss()

    if isinstance(train_set, Dataset):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
        assert isinstance(learning_rate, float)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        assert isinstance(learning_rate, List)
        assert len(learning_rate) == len(train_set)

        train_loader = [
            torch.utils.data.DataLoader(train, batch_size=100) for train in train_set
        ]
        optimizer = [
            torch.optim.SGD(
                model.parameters(), lr=learn_rate, weight_decay=weight_decay
            )
            for learn_rate in learning_rate
        ]

    test_accuracy_list, train_loss_list, iteration_list = train(
        num_epochs=num_epochs,
        train_loaders=train_loader,
        test_loader=test_loader,
        model=model,
        error=error,
        optimizers=optimizer,
    )
    # save_model(model, model_name, test_accuracy_list, train_loss_list, iteration_list)

    return model, test_accuracy_list, train_loss_list, iteration_list


def save_model(
    model,
    name,
    test_accuracy_list,
    train_loss_list,
    iteration_list,
    model_name="model",
):
    os.makedirs(f"record/{name}/figs", exist_ok=True)
    torch.save(model, f"record/{name}/{model_name}.pth")
    np.save(f"record/{name}/{model_name}-test_acc.npy", np.array(test_accuracy_list))
    np.save(f"record/{name}/{model_name}-train_loss.npy", np.array(train_loss_list))
    np.save(f"record/{name}/{model_name}-iters.npy", np.array(iteration_list))


def load_model(name, model_name="model"):
    model = torch.load(f"record/{name}/{model_name}.pth")
    test_acc = np.load(f"record/{name}/{model_name}-test_acc.npy")
    train_loss = np.load(f"record/{name}/{model_name}-train_loss.npy")
    iters = np.load(f"record/{name}/{model_name}-iters.npy")
    return model, test_acc, train_loss, iters


def get_model(name, train_set, test_set, num_epochs, model_name="model"):
    if os.path.exists(f"record/{name}/{model_name}.pth"):
        return load_model(name, model_name)
    else:
        model, test_accuracy_list, train_loss_list, iteration_list = train_new_model(
            train_set=train_set, test_set=test_set, num_epochs=num_epochs
        )
        save_model(
            model,
            name=name,
            test_accuracy_list=test_accuracy_list,
            train_loss_list=train_loss_list,
            iteration_list=iteration_list,
            model_name=model_name,
        )
        return model, test_accuracy_list, train_loss_list, iteration_list
