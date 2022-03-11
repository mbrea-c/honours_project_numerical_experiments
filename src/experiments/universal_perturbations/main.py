import logging
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import Tensor
from experiments.deepfool import deepfool


def proj_lp(v, proj_radius):
    """Project on the l2 ball centered at 0 and of radius proj_radius"""
    v *= min(1, proj_radius / torch.linalg.norm(v))

    return v


def get_fooling_rate(dataset: Dataset, net, v, image_dim=(28 * 28,)):
    batch_size = 100
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    batch_v = v.repeat((batch_size, 1))

    total = 0
    correct = 0

    for images, labels in loader:
        test_orig = images.view(-1, *image_dim)
        outputs_orig = net(test_orig)
        predictions_orig = torch.max(outputs_orig, 1)[1]  # .to(device)

        test_pert = (images.view(-1, *image_dim) + batch_v).to(torch.float32)
        outputs_pert = net(test_pert)
        predictions_pert = torch.max(outputs_pert, 1)[1]  # .to(device)

        correct += (predictions_orig == predictions_pert).sum()

        total += labels.shape[0]

    fooling_rate = 1 - correct / total

    return fooling_rate


def predict_from_image(image: Tensor, net, image_dim=(28 * 28,)):
    image = image.view(-1, *image_dim)
    output = net(image)
    prediction = torch.max(output, 1)[1][0]

    return prediction


def universal_perturbation(
    dataset: Dataset,
    net,
    max_iter=10,
    max_iter_deepfool=50,
    delta=0.2,
    proj_radius=10,
    image_dim=(28 * 28,),
):
    v = torch.zeros(size=(1, *image_dim))
    iter = 0
    fooling_rate = get_fooling_rate(dataset, net, v, image_dim)
    logging.info(f"The initial fooling rate is {fooling_rate} (should be 0)")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    while fooling_rate <= 1 - delta and iter < max_iter:
        for image, label in loader:
            image = image.view(-1, *image_dim)
            # Conversion to float32 is necessary, since addition sometimes
            # returns a different dtype than the summands'
            image_pert = (image + v).to(torch.float32)
            if predict_from_image(image, net) == predict_from_image(image_pert, net):
                r, loop_i, k_0, k_i, x_i = deepfool(
                    image_pert, net, max_iter=max_iter_deepfool
                )
                v = proj_lp(v + r, proj_radius)

        fooling_rate = get_fooling_rate(dataset, net, v, image_dim)
        iter += 1
        logging.info(
            f"Iter {iter} / {max_iter}; Current fooling rate is {fooling_rate}"
        )
    return v
