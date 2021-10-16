import jax
from jax._src.numpy.lax_numpy import linspace
import jax.numpy as jnp
import numpy as np
import typing as typ


Array = typ.Union[float, jnp.ndarray]


def calc_backend():
    # Demo of changed behavior for non-functional code:
    # https://jax.readthedocs.io/en/latest/faq.html#jit-changes-the-behavior-of-my-function
    square_cpu = jax.jit(square, backend="cpu")
    square_gpu = jax.jit(square, backend="gpu")
    x = linspace(-1, 1, int(1e8))
    print("go")
    print(square_cpu(x).mean())
    print(square_gpu(x).mean())


def calc_grad():
    square_grad = jax.grad(lambda x: square(x).sum())
    # square_grad = lambda x: jnp.diag(jax.jacobian(square)(x))
    # square_grad = lambda x: jax.jvp(square, [x], [jnp.ones_like(x)])[1]
    x = jnp.linspace(-1, 1, 5)
    y = jnp.vstack([x, square(x), square_grad(x)]).T
    print(y)


# @jax.jit
def square(x: Array) -> Array:
    return x ** 2
    # return jnp.power(x, 2)


def main():
    # calc_backend()
    calc_grad()
    # print(optimize(fun=square, x=4.0))


def optimize(*, fun: typ.Callable[[Array], float], x: Array) -> Array:
    fun_grad = jax.grad(fun)
    rate = 0.4
    nsteps = 20
    for _ in range(nsteps):
        x -= rate * fun_grad(x)
        print(x)
    return x


if __name__ == "__main__":
    main()
