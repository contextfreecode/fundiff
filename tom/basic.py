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
    quad_cpu = jax.jit(quad, backend="cpu")
    quad_gpu = jax.jit(quad, backend="gpu")
    x = linspace(-1, 1, int(1e8))
    # TODO Time execution.
    print(quad_cpu(x).shape)
    print(quad_gpu(x).shape)


def calc_grad():
    quad_grad = jax.grad(lambda x: quad(x).sum())
    quad_grad2 = lambda x: np.diag(jax.jacfwd(quad)(x))
    x = np.linspace(-4, 4, 17)
    y = np.vstack([x, quad(x), quad_grad(x), quad_grad2(x)])
    print(y.T)


# @jax.jit
def quad(x: Array) -> Array:
    return x ** 2


def main():
    calc_backend()
    calc_grad()


main()
