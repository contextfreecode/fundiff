import jax
from jax._src.numpy.lax_numpy import linspace
import jax.numpy as np
# import numpy as np
import typing as typ


Array = typ.Union[float, np.ndarray]


def calc_backend():
    # TODO At what function complexity does jit matter or not?
    # TODO Demo of changed behavior for non-functional code:
    # TODO https://jax.readthedocs.io/en/latest/faq.html#jit-changes-the-behavior-of-my-function
    square_cpu = jax.jit(square, backend="cpu")
    square_gpu = jax.jit(square, backend="gpu")
    x = linspace(-1, 1, int(1e8))
    print("go")
    # TODO Time execution.
    print(square_cpu(x).mean())
    print(square_gpu(x).mean())


def calc_grad():
    square_grad = jax.grad(lambda x: square(x).sum())
    square_grad2 = lambda x: np.diag(jax.jacfwd(square)(x))
    x = np.linspace(-4, 4, 17)
    y = np.vstack([x, square(x), square_grad(x), square_grad2(x)]).T
    print(y)
    print(np.vstack(jax.jvp(square, [x], [np.ones_like(x)])).T)


# @jax.jit
def square(x: Array) -> Array:
    return x ** 2


def main():
    calc_backend()
    calc_grad()


main()
