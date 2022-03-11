import mnist_torch
import logging
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="[{levelname}({name}):{filename}:{funcName}] {message}",
    )

    set_start_method("spawn")

    mnist_torch.run()
