import numpy as np
from torch import Tensor
import torch
import collections.abc


def predict(one_hot_label: Tensor):
    return one_hot_label.argmax()


def zero_gradients(x):
    """
    This function was removed from PyTorch,
    implementing it here manually for convenience
    """
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def deepfool(image, net, num_classes=10, max_iter=50, overshoot=0.02):
    image = image.detach().clone()
    image.requires_grad_()
    image.retain_grad()
    f_image = net.forward(image)
    k_0 = predict(f_image)

    x_i = image
    x_i.requires_grad_()
    input_shape = image.shape

    loop_i = 0

    w = torch.zeros(input_shape)
    r = torch.zeros(input_shape)

    fs = net.forward(x_i)
    k_i = k_0

    while k_i == k_0 and loop_i < max_iter:
        pert = np.inf
        fs[0, k_0].backward(retain_graph=True)
        grad_k_0 = x_i.grad.data.detach().clone()

        for k in [i for i in range(num_classes) if i != k_0]:
            zero_gradients(x_i)
            fs[0, k].backward(retain_graph=True)
            grad_k = x_i.grad.data.detach()

            w_k = grad_k - grad_k_0
            f_k = (fs[0, k] - fs[0, k_0]).data.detach()

            pert_k = -f_k / torch.linalg.vector_norm(w_k)
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = pert * w / torch.linalg.vector_norm(w)
        r += r_i

        x_i = (image + (1 + overshoot) * r).detach()
        x_i.requires_grad_()

        fs = net.forward(x_i)
        k_i = predict(fs)

        loop_i += 1

    r = (1 + overshoot) * r

    return r, loop_i, k_0, k_i, x_i
