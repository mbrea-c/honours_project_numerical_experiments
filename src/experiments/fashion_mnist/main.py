import numpy as np
import matplotlib.pyplot as plt
import os.path
import logging
from common.sgd import sgd_minibatch
from common.logistic_regression import LogisticRegression
from common.neural_network import NeuralNetwork, Softmax, Sigmoid, CrossEntropy, L2

DATASET_PATH = os.path.expandvars("${HOME}/datasets/fashion_mnist")


def load_dataset(dataset_path, val_frac=0.2):
    logging.info(f"Loading dataset at {dataset_path}")
    train = np.loadtxt(
        f"{dataset_path}/fashion-mnist_train.csv", delimiter=",", dtype=int, skiprows=1
    )
    test = np.loadtxt(
        f"{dataset_path}/fashion-mnist_test.csv", delimiter=",", dtype=int, skiprows=1
    )

    n_train = int(train.shape[0] * (1 - val_frac))

    # Split training data into train and validation sets
    train, val = np.split(train, [n_train], axis=0)

    y_train, x_train = np.split(train, [1], axis=1)
    y_val, x_val = np.split(val, [1], axis=1)
    y_test, x_test = np.split(test, [1], axis=1)

    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"x_train shape: {x_train.shape}")
    logging.info(f"y_val shape: {y_val.shape}")
    logging.info(f"x_val shape: {x_val.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    logging.info(f"x_test shape: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocessing(x_train, y_train, x_val, y_val, x_test, y_test):
    y_train = one_hot_encode(y_train, 10)
    y_val = one_hot_encode(y_val, 10)
    y_test = one_hot_encode(y_test, 10)

    x_train = x_train.astype(np.float64)
    x_val = x_val.astype(np.float64)
    x_test = x_test.astype(np.float64)

    std = np.std(x_train, axis=0).reshape((1, -1))
    mean = np.mean(x_train, axis=0).reshape((1, -1))

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_val, y_val, x_test, y_test


def one_hot_encode(labels, n_classes):
    n_examples = np.size(labels)
    labels_encoded = np.zeros((n_examples, n_classes))
    for i in range(n_examples):
        labels_encoded[i, labels[i]] = 1
    return labels_encoded


def plot_random_samples(x_train, y_train):
    logging.info(f"Plotting random samples")
    labels = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    axes = axes.flatten()
    idx = np.random.randint(0, 42000, size=10)
    for i in range(4):
        axes[i].imshow(x_train[idx[i], :].reshape(28, 28), cmap="gray")
        axes[i].axis("off")  # hide the axes ticks
        axes[i].set_title(labels[int(y_train[idx[i]])], color="black", fontsize=25)
    plt.savefig(
        "figs/samples_from_fashion_mnist.pdf", bbox_inches="tight", pad_inches=0
    )


def run():
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(DATASET_PATH)
    plot_random_samples(x_train, y_train)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    logistic_regression_model = LogisticRegression(
        weights_init=np.zeros((10, x_train.shape[1]), dtype=np.float64)
    )

    neural_network_model = NeuralNetwork(
        input_dim=x_train.shape[1],
        layer_params=[
            (1000, Sigmoid()),
            (200, Sigmoid()),
            (y_train.shape[1], Softmax()),
        ],
        cost_fn=CrossEntropy(),
    )

    sgd_minibatch(
        neural_network_model,
        x_train,
        y_train,
        x_val,
        y_val,
        patience=8,
        learning_rates=[1, 0.1, 0.03, 0.01],
        batch_size=64,
    )

    nn_accuracy = neural_network_model.pred_accuracy(x_val, y_val)
    logging.info(f"The accuracy of the nn model is {nn_accuracy}")

    # sgd_minibatch(logistic_regression_model, x_train, y_train, x_val, y_val)

    # logreg_accuracy = logistic_regression_model.pred_accuracy(x_val, y_val)
    # logging.info(f"The accuracy of the model is {logreg_accuracy}")
