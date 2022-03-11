import os
import logging
from datetime import datetime
from typing import Callable, List, Optional, Tuple
from torch.utils.data.dataset import ChainDataset, IterableDataset
from PIL import Image
import torchvision
import torchvision.transforms.functional
import torch.utils.data
import torch.optim
from torchvision import transforms
import torch
import pandas as pd
from torch import nn, Tensor, LongTensor
from torch.utils.data import Dataset, TensorDataset
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from experiments.deepfool import deepfool
from experiments.fashion_mnist_torch.model import (
    get_accuracy,
    get_model,
    load_model,
    save_model,
    train_more_epochs,
    train_new_model,
)
from experiments.formal_bound.main import lower_bound
from experiments.universal_perturbations.main import universal_perturbation
from torch.multiprocessing import Pool, set_start_method
from imgaug import augmenters as iaa
import numpy as np


def load_dataset():
    logging.info(f"Getting dataset")
    DATASET_DIR = os.path.expandvars("${HOME}/datasets/pytorch")

    train_set = torchvision.datasets.MNIST(
        DATASET_DIR,
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test_set = torchvision.datasets.MNIST(
        DATASET_DIR,
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    return train_set, test_set


class EagerDataset(Dataset):
    def __init__(self, subset: Dataset, transform: Callable):
        images_sequence = []
        count = 0
        for img, _ in subset:
            images_sequence.append(transform(img).detach().clone())
            if count % 500 == 0:
                logging.info(f"Transformed {count} images")
            count += 1
        images = torch.stack(images_sequence)
        labels = LongTensor([label for _, label in subset]).clone().detach()
        self.subset = TensorDataset(images, labels)

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.subset)


class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class UnionDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        base = 0
        for dataset in self.datasets:
            if index < base + len(dataset):
                return dataset[index - base]
            base += len(dataset)
        raise IndexError(f"Index {index} out of bounds")

    def __len__(self):
        return sum(map(lambda d: len(d), self.datasets))


def random_subset(dataset: Dataset, subset_size: int, n_items: int):
    indices = list(np.random.choice(n_items, size=(subset_size,), replace=False))
    return torch.utils.data.Subset(dataset, indices)


def output_label_fashion_mnist(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    input = label.item() if type(label) == Tensor else label
    return output_mapping[input]


def output_label_mnist(label):
    input = label.item() if type(label) == Tensor else label
    return input


def output_label(label):
    return output_label_mnist(label)


def predict_from_image(image: Tensor, net, image_dim=(28 * 28,)) -> int:
    image = image.view(-1, *image_dim)
    output = net(image)
    prediction = torch.max(output, 1)[1][0].item()
    assert isinstance(prediction, int)
    return prediction


def plot_random_samples(train_set, name):
    logging.info(f"Plotting random samples")

    subplots = (2, 5)
    n_samples = subplots[0] * subplots[1]

    _, axes = plt.subplots(*subplots, figsize=(12, 5))
    axes = axes.flatten()
    demo_loader = torch.utils.data.DataLoader(
        train_set, batch_size=n_samples, shuffle=True
    )
    images, labels = iter(demo_loader).next()
    for i in range(n_samples):
        image, label = images[i], labels[i]
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].axis("off")  # hide the axes ticks
        axes[i].set_title(output_label(label), color="black", fontsize=25)
    plt.savefig(
        f"record/{name}/figs/samples_from_fashion_mnist.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_side_to_side(
    images, name: str, figname: str = "deepfool_side_to_side.pdf", shape=(28, 28)
):
    logging.info(f"Plotting side to side")
    if isinstance(images[0], Tuple):
        images = [images]

    subplots = (len(images), len(images[0]))

    _, axes = plt.subplots(*subplots, figsize=(12, 5))
    if subplots[0] == 1:
        axes = [axes]
    for row in range(subplots[0]):
        for col in range(subplots[1]):
            image, label, params = images[row][col]
            label = f"{label}"
            vrange = params.get("vrange", (torch.min(image), torch.max(image)))
            cmap = params.get("cmap", "gray")
            axes[row][col].imshow(
                image.view(*shape).detach().numpy(),
                cmap=cmap,
                vmin=vrange[0],
                vmax=vrange[1],
            )
            axes[row][col].axis("off")  # hide the axes ticks
            if len(label):
                axes[row][col].set_title(f"{label}", color="black", fontsize=13)
        plt.savefig(f"record/{name}/figs/{figname}", bbox_inches="tight", pad_inches=0)


def plot_misclassification_matrix(matrix, name, figname="misclassification_matrix.pdf"):
    _, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(matrix, cmap="Blues", alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va="center", ha="center", size="xx-large")
    plt.xlabel("Prediction after perturbation", fontsize=18)
    plt.ylabel("Prediction before perturbation", fontsize=18)
    # plt.title("Confusion Matrix", fontsize=18)
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 10))
    plt.savefig(f"record/{name}/figs/{figname}")


def plot_against_universal_perturbation(train_set, net, v, name):
    logging.info(f"Plotting random samples against universal perturbation")

    subplots = (2, 5)
    n_samples = subplots[1]

    _, axes = plt.subplots(*subplots, figsize=(12, 5))
    demo_loader = torch.utils.data.DataLoader(
        train_set, batch_size=n_samples, shuffle=True
    )
    images, _ = iter(demo_loader).next()
    for i in range(n_samples):
        image = images[i].view(-1, 28 * 28)
        pred = predict_from_image(image, net)
        image_pert = (image + v).to(torch.float32)
        pred_pert = predict_from_image(image_pert, net)
        axes[0, i].imshow(image.view(28, 28), cmap="gray")
        axes[0, i].axis("off")  # hide the axes ticks
        axes[0, i].set_title(output_label(pred), color="black", fontsize=25)

        axes[1, i].imshow(image_pert.view(28, 28), cmap="gray")
        axes[1, i].axis("off")  # hide the axes ticks
        axes[1, i].set_title(output_label(pred_pert), color="black", fontsize=25)
    plt.savefig(
        f"record/{name}/figs/universal_perturbation.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_performance(name, model_name):

    model, test_accuracy_list, train_loss_list, iteration_list = load_model(
        name=name, model_name=model_name
    )

    plt.clf()
    plt.figure(figsize=(4, 4))
    plt.plot(iteration_list, train_loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.savefig(
        f"record/{name}/figs/nn_fashion_mnist_loss_vs_iters.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.clf()
    plt.figure(figsize=(4, 4))
    plt.plot(iteration_list, test_accuracy_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy")
    plt.savefig(
        f"record/{name}/figs/nn_fashion_mnist_acc_vs_iters.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    logging.info(
        f"Test performance is {test_accuracy_list[-1]} for {name}-{model_name} "
    )


def add_to_misclassification_matrix(y_true, y_pred, n_classes, matrix=None):
    if matrix is None:
        matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    for y in y_true:
        for yp in y_pred:
            matrix[y, yp] += 1
    return matrix


def experiment_deepfool_single_image(
    dataset,
    name: str,
    model_name: str = "model",
    figname="deepfool_side_to_side.pdf",
    index: Optional[int] = None,
):
    model, _, _, _ = load_model(name=name, model_name=model_name)

    if not index:
        index = int(torch.randint(0, len(dataset), size=()).item())
    image, label = dataset[index]
    image = image.view(-1, 28 * 28)

    r, loop_i, k_0, k_i, x_i = deepfool(image, model)
    logging.info(
        f"Deepfool: original pred {k_0} (true label {label}), new_pred {k_i}, iterations {loop_i}"
    )
    r_scaled, vrange = normalize_noise(r)
    plot_side_to_side(
        [
            (image, f"Original image (predicted {k_0})", {"vrange": (0, 1)}),
            (r, "Perturbation", {"vrange": (-1, 1), "cmap": "coolwarm"}),
            (r_scaled, "Perturbation (scaled)", {"vrange": vrange, "cmap": "coolwarm"}),
            (x_i, f"Perturbed image (predicted {k_i})", {"vrange": (0, 1)}),
        ],
        name=name,
        figname=figname,
    )


def experiment_deepfool_filter_single_image(
    dataset: Dataset, name: str, model_name: str = "model"
):
    model, _, _, _ = load_model(name=name, model_name=model_name)

    images, labels = iter(
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    ).next()
    image, label = images[0], labels[0]
    image = image.view(-1, 28 * 28)

    r, loop_i, k_0, k_i, x_i = deepfool(image, model)
    x_filter = filter_image(x_i.view(1, 28, 28))
    k_filter = predict_from_image(x_filter, model)
    logging.info(
        f"Deepfool: original pred {k_0} (true label {label}), new_pred {k_i}, after filter {k_filter} iterations {loop_i}"
    )
    plot_side_to_side(
        [(image, k_0, dict()), (x_i, k_i, dict()), (x_filter, k_filter, dict())],
        name=name,
        figname="deepfool_side_to_side_with_filter.pdf",
    )


def experiment_deepfool_jpeg_single_image(dataset: Dataset, name, model_name="model"):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    images, labels = iter(
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    ).next()
    image, label = images[0], labels[0]
    image = image.view(-1, 28 * 28)

    r, loop_i, k_0, k_i, x_i = deepfool(image, model)
    x_filter = jpeg_transform(x_i.view(28, 28, 1))
    k_filter = predict_from_image(x_filter, model)
    logging.info(
        f"Deepfool: original pred {k_0} (true label {label}), new_pred {k_i}, after filter {k_filter} iterations {loop_i}"
    )
    plot_side_to_side(
        [(image, k_0, dict()), (x_i, k_i, dict()), (x_filter, k_filter, dict())],
        name=name,
        figname="deepfool_side_to_side_with_jpeg.pdf",
    )


def run_deepfool_misc_matrix_experiment(
    dataset: Dataset, name, model_name="model", n_items=None
):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    logging.info("Running deepfool confusion matrix experiment")
    if not n_items:
        if isinstance(dataset, MNIST):
            n_items = len(dataset)
        else:
            logging.error(
                "Need to specify n_items if not using dataset of recognized type"
            )
            return
    subset_size = 1000
    subset = random_subset(dataset, subset_size=subset_size, n_items=n_items)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=True)
    matrix = None
    for i, (images, labels) in enumerate(iter(dataloader)):
        image, label = images[0], labels[0]
        image = image.view(-1, 28 * 28)
        r, loop_i, k_0, k_i, x_i = deepfool(image, model)
        matrix = add_to_misclassification_matrix(
            [k_0], [k_i], n_classes=10, matrix=matrix
        )
        if i % 50 == 0:
            logging.info(f"Iteration {i}/{subset_size}")
    plot_misclassification_matrix(
        matrix, name=name, figname="deepfool_confusion_matrix.pdf"
    )


def run_universal_perturbation_experiment(dataset, name, model_name="model"):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    logging.info("Finding universal perturbation")
    v = universal_perturbation(
        random_subset(dataset, 200, len(dataset)), model, proj_radius=3
    )
    plot_against_universal_perturbation(dataset, model, v, name=name)


def filter_image(image):
    image = torchvision.transforms.functional.gaussian_blur(image, 3)
    image = torchvision.transforms.functional.adjust_contrast(image, 2)

    return image


jpeg_augmenter = iaa.arithmetic.JpegCompression(compression=20)
jpeg_transform = transforms.Lambda(lambda x: jpeg_compress_image(x))
filter_transform = transforms.Lambda(lambda x: filter_image(x))


def jpeg_compress_image(image, target_shape=(28, 28, 1)):
    shape = image.shape
    dtype = image.type()
    min = torch.min(image)
    max = torch.max(image)
    image = image.view(*target_shape)
    image = (image - min) / (max - min)
    image = (255 * image).type(dtype=torch.uint8).numpy()
    image = jpeg_augmenter.augment_image(image)
    image = (torch.FloatTensor(image) / 255) * (max - min) + min
    image = torch.Tensor(image).type(dtype=dtype)
    image = image.view(*shape)
    return image


def fooling_rate(orig_dataset, transformed_dataset, model):
    dataloader_orig = torch.utils.data.DataLoader(
        orig_dataset, batch_size=1, shuffle=False
    )
    dataloader_transform = torch.utils.data.DataLoader(
        transformed_dataset, batch_size=1, shuffle=False
    )

    total = 0
    fooled = 0

    for orig, trans in zip(dataloader_orig, dataloader_transform):
        orig_x, _ = orig
        trans_x, _ = trans
        k_orig = predict_from_image(orig_x, model)
        k_trans = predict_from_image(trans_x, model)

        if k_orig != k_trans:
            fooled += 1
        total += 1

        # if total % 50 == 0:
        # logging.info(f"Fooling rate {fooled}/{total} ({100 * fooled / total}%)")

    return fooled / total


def normalize_noise(im):
    neg_scale = torch.max(torch.abs(torch.clamp(im, max=0)))
    pos_scale = torch.max(torch.abs(torch.clamp(im, min=0)))
    scale = max(pos_scale, neg_scale)
    return im / scale, (-1, 1)


def experiment_deepfool_filter_fooling_rate(
    dataset, name, model_name="model", subset_size=200
):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    subset = random_subset(dataset, subset_size=subset_size, n_items=len(dataset))

    p_adv = robustness(subset, model)
    random_transform = transforms.Lambda(lambda x: apply_random(x, p_adv))
    perturbed_subset = EagerDataset(
        subset,
        lambda x: apply_deepfool(x, model),
    )

    idx = int(torch.randint(0, subset_size, size=()).item())
    orig_img = subset[idx][0]
    orig_k = predict_from_image(orig_img, model)
    images = [
        [
            (orig_img, f"Original ({orig_k})", {"vrange": (0, 1)}),
        ],
        [
            (
                torch.zeros_like(orig_img),
                "",
                {"vrange": (-1, 1), "cmap": "coolwarm"},
            ),
        ],
    ]

    fooling_rates = []
    accuracy = []
    transform_names = ["Identity", "Gaussian", "Random", "JPEG"]
    transform_objs = [
        transforms.Compose([]),
        filter_transform,
        random_transform,
        jpeg_transform,
    ]

    for transform_name, transform in zip(transform_names, transform_objs):

        transformed_dataset = TransformedDataset(perturbed_subset, transform)
        transformed_original = TransformedDataset(subset, transform)
        img = transformed_dataset[idx][0]
        k = predict_from_image(img, model)
        images[0].append((img, f"DF+{transform_name} ({k})", {"vrange": (0, 1)}))
        diff, vrange = normalize_noise(img - orig_img)
        images[1].append((diff, "", {"vrange": vrange, "cmap": "coolwarm"}))
        logging.info(f"Computing fooling rate after {transform_name}...")
        fooling_rates.append(
            fooling_rate(
                orig_dataset=subset,
                transformed_dataset=transformed_dataset,
                model=model,
            )
        )
        accuracy.append(get_accuracy(transformed_original, model))
        logging.info(f"Fooling rate after {transform_name} is {fooling_rates[-1]}")

    df = pd.DataFrame(
        np.array(
            [fooling_rates, list(map(lambda a: f"{100*a:.1f}\\%", accuracy))]
        ).reshape((2, -1)),
        columns=transform_names,
        index=["\\(\\gamma_{fool}\\)", "Accuracy"],
    )
    with open(f"record/{name}/figs/df_image_filter_fooling_rates.tex", "w") as tf:
        tf.write(df.to_latex(escape=False))
    plot_side_to_side(
        images,
        name=name,
        figname="df_image_filters_side_to_side.pdf",
    )


def experiment_robustness_both_bounds(
    dataset,
    name,
    model_names=[
        "model",
        "model-deepfool_retrained",
        "model-random_retrained",
        "model-identity_retrained",
    ],
    subset_size=10,
):
    lower_bounds = []
    upper_bounds = []

    subset = random_subset(dataset, subset_size, len(dataset))

    for model_name in model_names:
        model, _, _, _ = load_model(name=name, model_name=model_name)

        logging.info(f"Computing bounds for {model_name}")

        upper_bounds.append(robustness(subset, model))
        lower_bounds.append(robustness_lower_bound(subset, model))

    df = pd.DataFrame(
        np.array([upper_bounds, lower_bounds]).reshape((2, -1)),
        columns=["Original", "Deepfool", "Random", "Clean"],
        index=[
            "\\(\\hat{\\rho}_{adv}\\) upper bound",
            "\\(\\hat{\\rho}_{adv}\\) lower bound",
        ],
    )
    with open(f"record/{name}/figs/upper_and_lower_bounds_retrained.tex", "w") as tf:
        tf.write(df.to_latex(escape=False))


def robustness(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    total = 0
    p_adv = 0
    for images, labels in dataloader:
        image, label = images[0], labels[0]
        image = image.view(-1, 28 * 28)
        r, loop_i, k_0, k_i, x_i = deepfool(image, model)
        r_norm = torch.linalg.vector_norm(torch.as_tensor(r))
        image_norm = torch.linalg.vector_norm(image)
        p_adv += r_norm / image_norm
        total += 1

        if total % 50 == 0:
            logging.info(f"Robustness {p_adv / total} on iter {total}")
    return p_adv / total


class DisgustingHelperObj:
    def __init__(self, model):
        self.model = model

    def __call__(self, params):
        img, k = params
        print("Starting computation of lower bound")
        r_norm, _ = lower_bound(model=self.model, k=k, x=img, r_range=[2])
        image_norm = torch.linalg.vector_norm(img)
        rel_norm = r_norm / image_norm
        print(f"Relative norm lower bound: {rel_norm}")
        return rel_norm


def robustness_lower_bound(dataset, model, n_workers=2):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    total = 0
    p_adv = 0

    call_obj = DisgustingHelperObj(model)

    list_images = []

    for images, labels in dataloader:
        image, _ = images[0], labels[0]
        k = predict_from_image(image, model)
        image = image.view(-1)
        list_images.append((image, k))
    with Pool(n_workers) as p:
        for rel_norm in p.map(call_obj, list_images):
            p_adv += rel_norm
            total += 1
    return p_adv / total


def experiment_deepfool_robustness(
    train_set,
    test_set,
    name,
    model_name="model",
    subset_size=300,
    lower_bound_subset_size=10,
    extension_size=5000,
    n_iters=8,
    always_run: bool = True,
):
    extension_subset = random_subset(
        train_set,
        subset_size=extension_size,
        n_items=len(train_set),
    )

    subset = random_subset(train_set, subset_size=subset_size, n_items=len(train_set))
    lower_bound_subset = random_subset(
        train_set, subset_size=lower_bound_subset_size, n_items=len(train_set)
    )

    def should_run(extension_name: str):
        return (
            always_run
            or not os.path.exists(f"record/{name}/{extension_name}-robustness_list.npy")
            or not os.path.exists(
                f"record/{name}/{extension_name}-robustness_lower_list.npy"
            )
            or not os.path.exists(
                f"record/{name}/{extension_name}-robustness_upper_list.npy"
            )
        )

    if any(map(should_run, ["deepfool", "random", "identity"])):
        model, _, _, _ = load_model(name=name, model_name=model_name)
        logging.info(f"Getting base measurements of model...")
        base_p_adv = robustness(subset, model)
        base_p_adv_upper = robustness(lower_bound_subset, model)
        base_p_adv_lower = robustness_lower_bound(lower_bound_subset, model)
        logging.info(f"{base_p_adv_lower} < robustness < {base_p_adv_upper}")

    def get_p_adv_for_extension(extension_name: str):
        if should_run(extension_name):
            model, _, _, _ = load_model(name=name, model_name=model_name)
            p_adv = [base_p_adv]
            p_adv_lower = [base_p_adv_lower]
            p_adv_upper = [base_p_adv_upper]

            logging.info(f"Computing {extension_name} extension dataset...")

            if extension_name == "deepfool":
                extension = EagerDataset(
                    extension_subset,
                    lambda x: apply_deepfool(x, model),
                )
            elif extension_name == "random":
                extension = EagerDataset(
                    extension_subset, lambda x: apply_random(x, p_adv[-1])
                )
            elif extension_name == "identity":
                extension = EagerDataset(extension_subset, lambda x: x)
            else:
                raise Exception(f"Extension {extension_name} not recognized")

            test_acc, train_loss, iters_list = [], [], []
            last_iter = 0
            for n in range(1, n_iters):
                logging.info(
                    f"Retraining model on {extension_name} extended dataset..."
                )
                _, n_test_acc, n_train_loss, n_iters_list = train_more_epochs(
                    [train_set, extension],
                    test_set,
                    model,
                    num_epochs=1,
                    learning_rate=[1e-2, (1e-2) / 2],
                )
                test_acc += n_test_acc
                train_loss += n_train_loss
                iters_list += list(map(lambda x: x + last_iter, n_iters_list))
                last_iter = iters_list[-1]

                logging.info(
                    f"Computing robustness of model after retraining {n}/{n_iters} epochs..."
                )
                p_adv.append(robustness(subset, model))
                logging.info(
                    f"Robustness is {p_adv[-1]} after retraining {n}/{n_iters} epochs"
                )
                logging.info(
                    f"Computing lower and upper bounds after retraining {n}/{n_iters} epochs..."
                )
                p_adv_upper.append(robustness(lower_bound_subset, model))
                p_adv_lower.append(robustness_lower_bound(lower_bound_subset, model))
                logging.info(f"{p_adv_lower[-1]} < robustness < {p_adv_upper[-1]}")

            save_model(
                model=model,
                name=name,
                model_name=f"{model_name}-{extension_name}_retrained",
                test_accuracy_list=test_acc,
                train_loss_list=train_loss,
                iteration_list=iters_list,
            )

            np.save(
                f"record/{name}/{extension_name}-robustness_list.npy", np.array(p_adv)
            )
            np.save(
                f"record/{name}/{extension_name}-robustness_lower_list.npy",
                np.array(p_adv_lower),
            )
            np.save(
                f"record/{name}/{extension_name}-robustness_upper_list.npy",
                np.array(p_adv_upper),
            )
        else:
            p_adv = np.load(f"record/{name}/{extension_name}-robustness_list.npy")
            p_adv_lower = np.load(
                f"record/{name}/{extension_name}-robustness_lower_list.npy"
            )
            p_adv_upper = np.load(
                f"record/{name}/{extension_name}-robustness_upper_list.npy"
            )
        return p_adv, p_adv_lower, p_adv_upper

    p_adv_df, p_adv_lower_df, p_adv_upper_df = get_p_adv_for_extension("deepfool")
    p_adv_rand, p_adv_lower_rand, p_adv_upper_rand = get_p_adv_for_extension("random")
    p_adv_id, p_adv_lower_id, p_adv_upper_id = get_p_adv_for_extension("identity")

    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.grid(visible=True, alpha=0.3)
    plt.plot(list(range(n_iters)), p_adv_df, label="DeepFool", linestyle="-")
    plt.plot(list(range(n_iters)), p_adv_rand, label="Random", linestyle="--")
    plt.plot(list(range(n_iters)), p_adv_id, label="Clean", linestyle=":")
    plt.legend()
    plt.xlabel("No. of epochs")
    plt.ylabel("Robustness")
    plt.savefig(
        f"record/{name}/figs/deepfool_robustness_vs_extra_epochs.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.clf()
    plt.figure(figsize=(4, 4))
    plt.grid(visible=True, alpha=0.3)
    plt.plot(list(range(n_iters)), p_adv_lower_df, label="DeepFool", linestyle="-")
    plt.plot(list(range(n_iters)), p_adv_lower_rand, label="Random", linestyle="--")
    plt.plot(list(range(n_iters)), p_adv_lower_id, label="Clean", linestyle=":")
    plt.legend()
    plt.xlabel("No. of epochs")
    plt.ylabel("Robustness (lower bound)")
    plt.savefig(
        f"record/{name}/figs/deepfool_robustness_lower_vs_extra_epochs.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.clf()
    plt.figure(figsize=(4, 4))
    plt.grid(visible=True, alpha=0.3)
    plt.plot(list(range(n_iters)), p_adv_upper_df, label="DeepFool", linestyle="-")
    plt.plot(list(range(n_iters)), p_adv_upper_rand, label="Random", linestyle="--")
    plt.plot(list(range(n_iters)), p_adv_upper_id, label="Clean", linestyle=":")
    plt.legend()
    plt.xlabel("No. of epochs")
    plt.ylabel("Robustness (upper bound)")
    plt.savefig(
        f"record/{name}/figs/deepfool_robustness_upper_vs_extra_epochs.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.clf()
    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax2 = plt.twinx()
    ax.set_ylabel("Robustness (lower bound)", color="red")
    ax2.set_ylabel("Robustness (upper bound)", color="blue")
    plt.grid(visible=True, alpha=0.3)
    ax.plot(
        list(range(n_iters)),
        p_adv_lower_df,
        label="DeepFool (lower)",
        linestyle="-",
        color="red",
    )
    ax.plot(
        list(range(n_iters)),
        p_adv_lower_rand,
        label="Random (lower)",
        linestyle="--",
        color="red",
    )
    ax.plot(
        list(range(n_iters)),
        p_adv_lower_id,
        label="Clean (lower)",
        linestyle=":",
        color="red",
    )
    ax2.plot(
        list(range(n_iters)),
        p_adv_upper_df,
        label="DeepFool (upper)",
        linestyle="-",
        color="blue",
    )
    ax2.plot(
        list(range(n_iters)),
        p_adv_upper_rand,
        label="Random (upper)",
        linestyle="--",
        color="blue",
    )
    ax2.plot(
        list(range(n_iters)),
        p_adv_upper_id,
        label="Clean (upper)",
        linestyle=":",
        color="blue",
    )
    plt.legend()
    plt.xlabel("No. of epochs")
    # plt.ylabel("Robustness")
    plt.savefig(
        f"record/{name}/figs/deepfool_robustness_mixed_vs_extra_epochs.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )


def experiment_robustness_lower_bound(
    train_set,
    name,
    model_name="model",
    subset_size=20,
):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    subset = random_subset(train_set, subset_size, len(train_set))
    p_adv_lower = robustness_lower_bound(subset, model)
    p_adv_upper = robustness(subset, model)

    logging.info(f"{p_adv_lower} < ρₐₔᵥ < {p_adv_upper}")


def apply_deepfool(image, model):
    shape = image.shape
    image = image.view(-1, 28 * 28)
    r, loop_i, k_0, k_i, x_i = deepfool(image, model)
    image = Tensor(x_i).view(*shape)
    return image


def apply_random(image, relnorm):
    shape = image.shape
    target_norm = torch.linalg.vector_norm(image) * relnorm
    noise = torch.rand(shape)
    noise = noise * (target_norm / torch.linalg.vector_norm(noise))
    image = image + noise
    image = torch.clamp(image, min=0, max=1)
    return image


def figure_out_brightness_thingy(dataset, name, model_name="model"):
    images, labels = next(
        iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True))
    )
    image = images[0]
    _, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(image.view(28, 28).detach().numpy(), cmap="gray")
    axes[0].axis("off")  # hide the axes ticks
    axes[0].set_title(f"Before transform", color="black", fontsize=25)
    image = torchvision.transforms.functional.gaussian_blur(image, 3)
    axes[1].imshow(image.view(28, 28).detach().numpy(), cmap="gray")
    axes[1].axis("off")  # hide the axes ticks
    axes[1].set_title(f"Blur", color="black", fontsize=25)
    image = torchvision.transforms.functional.adjust_contrast(image, 2)
    axes[2].imshow(image.view(28, 28).detach().numpy(), cmap="gray")
    axes[2].axis("off")  # hide the axes ticks
    axes[2].set_title(f"Boost contrast", color="black", fontsize=25)

    plt.savefig(f"record/{name}/figs/deepfool_image_filters.pdf")


def experiment_filters_to_cifar10(name):
    image = Image.open("places2.jpg")
    tensor = transforms.ToTensor()(image).permute([1, 2, 0])
    shape = tensor.shape
    image_gaussian = filter_image(tensor.permute([2, 0, 1]))
    image_jpeg = jpeg_compress_image(tensor, target_shape=shape)
    image_random = apply_random(tensor, torch.linalg.vector_norm(tensor) / 1000)

    plot_side_to_side(
        [
            (tensor, "Original", dict()),
            (image_gaussian.permute([1, 2, 0]), "Gaussian", dict()),
            (image_random, "Random", dict()),
            (image_jpeg, "JPEG", dict()),
        ],
        name=name,
        figname="places2_filters.pdf",
        shape=shape,
    )


def df_upper_bound(model, x: Tensor) -> float:
    image = x.view(-1, 28 * 28)
    r, loop_i, k_0, k_i, x_i = deepfool(image, model)
    r_norm = torch.linalg.vector_norm(torch.as_tensor(r))
    r_norm = r_norm.item()
    assert isinstance(r_norm, float)
    return r_norm


def experiment_minimal_pert_bounds_single_image(dataset, name, model_name="model"):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    index = int(torch.randint(0, len(dataset), size=()).item())
    img = dataset[index][0]
    upper = df_upper_bound(model, img)
    lower, _ = lower_bound(
        model=model, k=predict_from_image(img, model), x=img.view(-1), r_range=[upper]
    )
    logging.info(f"{lower} < Δ(x,k) < {upper}")


def test_lower_bound(dataset, name, model_name="model", subset_size=20):
    model, _, _, _ = load_model(name=name, model_name=model_name)
    subset = random_subset(dataset, subset_size, len(dataset))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=True)
    error_found = False
    for images, _ in iter(dataloader):
        img = images[0]
        upper = df_upper_bound(model, img)
        lower, _ = lower_bound(
            model=model, k=predict_from_image(img, model), x=img.view(-1), r_range=[2]
        )
        if lower > upper:
            logging.error(
                f"Found lower bound {lower} greater than upper bound {upper}!"
            )
            error_found = True
    if not error_found:
        logging.info(f"Did not find any cases where lower bound fails")
    else:
        logging.error("The bound fails!")


def run():
    train_set, test_set = load_dataset()
    # train_set = random_subset(train_set, subset_size=1000, n_items=len(train_set))
    num_epochs = 2

    # name = f"mnist-{datetime.now().isoformat()}"
    name = "mnist-2022-02-24T12:57:25.286970"
    # name = "target"

    model, test_acc, train_loss, iters = get_model(
        name=name,
        model_name="model",
        train_set=train_set,
        test_set=test_set,
        num_epochs=num_epochs,
    )

    # plot_random_samples(train_set, name=name)
    # figure_out_brightness_thingy(train_set, name=name)

    plot_performance(name=name, model_name="model")

    df_index = int(torch.randint(0, len(train_set), size=()).item())

    # img = train_set[df_index][0]
    # lower_bound(model=model, k=predict_from_image(img, model), x=img.view(-1))

    # experiment_deepfool_single_image(
    # train_set, name=name, model_name="model", index=df_index
    # )
    # experiment_filters_to_cifar10(name=name)
    # experiment_deepfool_filter_single_image(train_set, name=name, model_name="model")
    # experiment_deepfool_jpeg_single_image(train_set, name=name, model_name="model")
    experiment_deepfool_filter_fooling_rate(train_set, name=name, model_name="model")
    # test_lower_bound(dataset=test_set, name=name)
    # experiment_minimal_pert_bounds_single_image(dataset=train_set, name=name)
    # experiment_robustness_lower_bound(train_set=train_set, name=name)
    # experiment_deepfool_robustness(
    # train_set,
    # test_set,
    # name=name,
    # n_iters=8,
    # model_name="model",
    # lower_bound_subset_size=10,
    # always_run=False,
    # )
    # experiment_robustness_both_bounds(train_set, name=name)
    # run_deepfool_misc_matrix_experiment(train_set, model)
    # run_universal_perturbation_experiment(test_set, model)
