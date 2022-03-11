import fashion_mnist_torch
import formal_bound_sympy
import logging
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="[{levelname}({name}):{filename}:{funcName}] {message}",
    )

    set_start_method("spawn")

    fashion_mnist_torch.run()
    # formal_bound_sympy.run()
