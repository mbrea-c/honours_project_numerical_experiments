from functools import reduce
import logging
import math
from typing import Iterable

import torch
from torch.functional import Tensor
from experiments.fashion_mnist_torch.model import FashionNN


class State:
    def __init__(self, model):
        self.max_act = dict()
        self.min_act = dict()
        self.max_grad_l = dict()
        self.min_grad_l = dict()
        self.max_sum_grad = dict()
        self.min_sum_grad = dict()
        self.beta_vals = dict()
        self.activation = model.activation(**model.activation_params)


def lower_bound(
    model: FashionNN, k: int, x: Tensor, r_range: Iterable[float] = range(2, 10, 2)
):
    per_r = []
    for R in r_range:
        per_r.append(lower_bound_r(model=model, R=R, k=k, x=x))
    result = max(per_r)
    logging.debug(f"Δ(x,k) ≥ {result}, chosen out of {per_r}")
    return result, per_r


def lower_bound_r(model: FashionNN, R: float, k: int, x: Tensor):
    state = State(model)
    n_classes = model.layers[-1][0].weight.shape[0]
    per_class = []
    for j in range(n_classes):
        if j != k:
            logging.debug(f"Getting bound for local lipschitz constant, k={k}, j={j}")
            local_cross_lipschitz = get_max_cross_lipschitz(
                model=model, j=j, k=k, R=R, x=x, state=state
            )
            fwd = model.forward(x)
            diff = fwd[k] - fwd[j]
            diff = diff.item()
            assert isinstance(diff, float)
            per_class.append(diff / local_cross_lipschitz)
    bound = min([min(per_class), R])
    logging.debug(f"Δ(x,k) ≥ {bound}, R={R}")
    return bound


def get_max_cross_lipschitz(
    model: FashionNN, j: int, k: int, R: float, x: Tensor, state: State
) -> float:
    n_layers = len(model.layers)
    n = n_layers - 1
    n_items, _ = model.layers[n - 1][0].weight.shape
    n_input, _ = model.layers[0][0].weight.shape
    total = 0

    def beta(r, u):
        if (j, k, r, u) in state.beta_vals:
            return state.beta_vals[(j, k, r, u)]
        else:
            w_jr = get_weight(model, n, j, r)
            w_kr = get_weight(model, n, k, r)
            w_ju = get_weight(model, n, j, u)
            w_ku = get_weight(model, n, k, u)
            beta_val = (w_jr - w_kr) * (w_ju - w_ku)
            state.beta_vals[(j, k, r, u)] = beta_val
            return beta_val

    for r in range(n_items):
        for u in range(n_items):
            beta_ru = beta(r, u)
            if beta_ru >= 0:
                max_act_r = get_max_act(
                    model=model, layer=n - 1, elem=r, R=R, x=x, state=state
                )
                max_act_u = get_max_act(
                    model=model, layer=n - 1, elem=u, R=R, x=x, state=state
                )
                max_act_r_grad = activation_grad(state=state, x=max_act_r)
                max_act_u_grad = activation_grad(state=state, x=max_act_u)
                total += (
                    beta_ru
                    * max_act_r_grad
                    * max_act_u_grad
                    * get_max_sum_grad_l(
                        model=model, r=r, u=u, layer=n - 1, R=R, x=x, state=state
                    )
                )
            else:
                min_act_r = get_min_act(
                    model=model, layer=n - 1, elem=r, R=R, x=x, state=state
                )
                min_act_u = get_min_act(
                    model=model, layer=n - 1, elem=u, R=R, x=x, state=state
                )
                min_act_r_grad = activation_grad(state=state, x=min_act_r)
                min_act_u_grad = activation_grad(state=state, x=min_act_u)
                total += (
                    beta_ru
                    * min_act_r_grad
                    * min_act_u_grad
                    * get_min_sum_grad_l(
                        model=model, r=r, u=u, layer=n - 1, R=R, x=x, state=state
                    )
                )
    return math.sqrt(total)


def activation(state: State, x: float) -> float:
    act = state.activation
    x_torch = Tensor([x])
    a = act(x_torch)
    a_f = a.item()
    assert isinstance(a_f, float)
    return a_f


def activation_grad(state: State, x: float) -> float:
    act = state.activation
    x_torch = torch.Tensor([x])
    x_torch.requires_grad_()
    act.zero_grad()
    act.forward(x_torch).backward()
    grad = x_torch.grad
    assert grad is not None
    grad_f = grad.item()
    assert isinstance(grad_f, float)
    return grad_f


def get_max_sum_grad_l(
    model: FashionNN, r: int, u: int, layer: int, R: float, x: Tensor, state: State
) -> float:
    if (r, u, layer) in state.max_sum_grad:
        return state.max_sum_grad[(r, u, layer)]
    n = model.layers[0][0].weight.shape[1]
    assert isinstance(n, int)

    max_r = get_max_grad_l(model=model, elem=r, layer=layer, R=R, x=x, state=state)
    max_u = get_max_grad_l(model=model, elem=u, layer=layer, R=R, x=x, state=state)
    min_r = get_min_grad_l(model=model, elem=r, layer=layer, R=R, x=x, state=state)
    min_u = get_min_grad_l(model=model, elem=u, layer=layer, R=R, x=x, state=state)

    max_max = max_r * max_u
    max_min = max_r * min_u
    min_max = min_r * max_u
    min_min = min_r * min_u

    maxer = reduce(torch.maximum, [max_max, max_min, min_max, min_min])
    result = torch.sum(maxer).item()
    assert isinstance(result, float)
    state.max_sum_grad[(r, u, layer)] = result
    return result


def get_min_sum_grad_l(
    model: FashionNN, r: int, u: int, layer: int, R: float, x: Tensor, state: State
) -> float:
    if (r, u, layer) in state.min_sum_grad:
        return state.min_sum_grad[(r, u, layer)]
    n = model.layers[0][0].weight.shape[1]
    assert isinstance(n, int)

    max_r = get_max_grad_l(model=model, elem=r, layer=layer, R=R, x=x, state=state)
    max_u = get_max_grad_l(model=model, elem=u, layer=layer, R=R, x=x, state=state)
    min_r = get_min_grad_l(model=model, elem=r, layer=layer, R=R, x=x, state=state)
    min_u = get_min_grad_l(model=model, elem=u, layer=layer, R=R, x=x, state=state)

    max_max = max_r * max_u
    max_min = max_r * min_u
    min_max = min_r * max_u
    min_min = min_r * min_u

    maxer = reduce(torch.minimum, [max_max, max_min, min_max, min_min])

    result = torch.sum(maxer).item()
    assert isinstance(result, float)
    state.min_sum_grad[(r, u, layer)] = result
    return result


def get_max_grad_l(
    model: FashionNN, elem: int, layer: int, R: float, x: Tensor, state: State
) -> Tensor:
    if (elem, layer) in state.max_grad_l:
        return state.max_grad_l[(elem, layer)]
    elif layer == 0:
        return model.layers[layer][0].weight[elem, :]
    else:
        _, n_items = model.layers[layer][0].weight.shape
        total = 0
        for i in range(n_items):
            w_ji = get_weight(model, layer, elem, i)
            if w_ji >= 0:
                max_act_i = get_max_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                max_act_i_grad = activation_grad(state=state, x=max_act_i)
                max_grad_il = get_max_grad_l(
                    model=model, elem=i, layer=layer - 1, R=R, x=x, state=state
                )
                total += w_ji * max_act_i_grad * max_grad_il
            else:
                min_act_i = get_min_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                min_act_i_grad = activation_grad(state=state, x=min_act_i)
                min_grad_il = get_min_grad_l(
                    model=model, elem=i, layer=layer - 1, R=R, x=x, state=state
                )
                total += w_ji * min_act_i_grad * min_grad_il
        state.max_grad_l[(elem, layer)] = total
        return total


def get_min_grad_l(
    model: FashionNN, elem: int, layer: int, R: float, x: Tensor, state: State
):
    if (elem, layer) in state.min_grad_l:
        return state.min_grad_l[(elem, layer)]
    elif layer == 0:
        return model.layers[layer][0].weight[elem, :]
    else:
        _, n_items = model.layers[layer][0].weight.shape
        total = 0
        for i in range(n_items):
            w_ji = get_weight(model, layer, elem, i)
            if w_ji < 0:
                max_act_i = get_max_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                max_act_i_grad = activation_grad(state=state, x=max_act_i)
                max_grad_il = get_max_grad_l(
                    model=model, elem=i, layer=layer - 1, R=R, x=x, state=state
                )
                total += w_ji * max_act_i_grad * max_grad_il
            else:
                min_act_i = get_min_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                min_act_i_grad = activation_grad(state=state, x=min_act_i)
                min_grad_il = get_min_grad_l(
                    model=model, elem=i, layer=layer - 1, R=R, x=x, state=state
                )
                total += w_ji * min_act_i_grad * min_grad_il
        state.min_grad_l[(elem, layer)] = total
        return total


def get_max_act(
    model: FashionNN, layer: int, elem: int, R: float, x: Tensor, state: State
):
    if (elem, layer) in state.max_act:
        return state.max_act[(elem, layer)]
    elif layer == 0:
        weight_vec = model.layers[layer][0].weight[elem, :]
        bias = get_bias(model, layer, elem)
        return (
            torch.inner(weight_vec, x) + R * torch.linalg.vector_norm(weight_vec) + bias
        ).item()
    else:
        _, n_items = model.layers[layer][0].weight.shape
        bias = get_bias(model, layer, elem)
        total = bias
        for i in range(n_items):
            w_ji = get_weight(model, layer, elem, i)
            if w_ji >= 0:
                max_act = get_max_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                total += w_ji * activation(state=state, x=max_act)
            else:
                min_act = get_min_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                total += w_ji * activation(state=state, x=min_act)
        state.max_act[(elem, layer)] = total
        return total


def get_min_act(
    model: FashionNN, layer: int, elem: int, R: float, x: Tensor, state: State
) -> float:
    if (elem, layer) in state.min_act:
        return state.min_act[(elem, layer)]
    elif layer == 0:
        weight_vec = model.layers[layer][0].weight[elem, :]
        bias = get_bias(model, layer, elem)
        return (
            torch.inner(weight_vec, x) - R * torch.linalg.vector_norm(weight_vec) + bias
        ).item()
    else:
        _, n_items = model.layers[layer][0].weight.shape
        bias = get_bias(model, layer, elem)
        total = bias
        for i in range(n_items):
            w_ji = get_weight(model, layer, elem, i)
            if w_ji < 0:
                max_act = get_max_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                total += w_ji * activation(state=state, x=max_act)
            else:
                min_act = get_min_act(
                    model=model, layer=layer - 1, elem=i, R=R, x=x, state=state
                )
                total += w_ji * activation(state=state, x=min_act)
        state.min_act[(elem, layer)] = total
        return total


def get_weight(model, layer, i, j):
    w = model.layers[layer][0].weight[i, j].item()
    assert isinstance(w, float)
    return w


def get_bias(model, layer, i):
    b = model.layers[layer][0].bias[i].item()
    assert isinstance(b, float)
    return b
