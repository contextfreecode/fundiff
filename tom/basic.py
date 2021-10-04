import jax
import jax.numpy as np
# import numpy as np
import typing as typ


ArrayLike = typ.Union[float, np.ndarray]


# TODO Benchmark with and without jit. (And cpu vs gpu.)
@jax.jit
def quad(x: ArrayLike) -> ArrayLike:
    return x ** 2


def main():
    quad_grad = jax.grad(lambda x: quad(x).sum())
    x = np.linspace(-4, 4, 17)
    y = np.vstack([x, quad(x), quad_grad(x)])
    print(y.T)


main()
