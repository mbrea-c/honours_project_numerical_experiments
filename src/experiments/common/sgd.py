import numpy as np
import logging
from common import Model


def sgd_minibatch(
    model: Model,
    x_train,
    y_train,
    x_val,
    y_val,
    learning_rates=[0.1, 0.01],
    batch_size=32,
    patience=25,
    batches_per_check=5,
):
    logging.info(
        f"Training with SGD: batch_size = {batch_size}, learning_rates = {learning_rates}, patience = {patience}"
    )

    best_cost = None
    best_params = None
    for learning_rate in learning_rates:
        logging.info(f"Setting learning_rate {learning_rate}")
        if best_cost is not None and best_params is not None:
            model.set_weights(best_params)
        checks_without_improvement = 0
        iters = 0
        while checks_without_improvement < patience:
            sgd_minibatch_iter(x_train, y_train, model, learning_rate, batch_size)
            iters += 1
            if iters % batches_per_check == 0:
                cost = model.cost(x_val, y_val)
                accuracy = model.pred_accuracy(x_val, y_val)
                checks_without_improvement += 1
                if best_cost is None or best_cost > cost:
                    checks_without_improvement = 0
                    best_cost = cost
                    best_params = (
                        model.get_weights()
                    )  # model.get_weights returns *COPY* of the weights
                logging.info(
                    f"[Iter {iters}] Current accuracy {accuracy}, cost {cost} vs best seen cost {best_cost}. Patience left {patience - checks_without_improvement}"
                )
    return best_params


def sgd_minibatch_iter(x_train, y_train, model: Model, learning_rate, batch_size):
    batch_indices = np.random.choice(x_train.shape[0], batch_size, replace=False)
    batch_x = x_train[batch_indices, :]
    batch_y = y_train[batch_indices, :]
    _ = model.forward(batch_x)
    model.backward(batch_y, learning_rate)
