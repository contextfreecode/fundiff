import jax
from jax._src.numpy.lax_numpy import linspace
import jax.numpy as jnp
import numpy as np
import typing as typ


Array = typ.Union[float, jnp.ndarray]


def sum_square(x: Array) -> Array:
    return jnp.sum(square(x))


# @jax.jit
def calc_speed():
    x = linspace(-1, 1, int(1e8))
    square_grad = jax.grad(sum_square)
    y = jnp.array([square(x), square_grad(x)])
    return y.mean(axis=1)


calc_speed_cpu = jax.jit(calc_speed, backend="cpu")
calc_speed_gpu = jax.jit(calc_speed, backend="gpu")


def calc_grad():
    square_grad = jax.grad(lambda x: square(x).sum())
    # square_grad = lambda x: jnp.diag(jax.jacobian(square)(x))
    # square_grad = lambda x: jax.jvp(square, [x], [jnp.ones_like(x)])[1]
    x = jnp.linspace(-1, 1, 5)
    y = jnp.vstack([x, square(x), square_grad(x)]).T
    print(y)


def square(x: Array) -> Array:
    return x ** 2
    # return jnp.power(x, 2)
    # return x ** 4 - 2 * x ** 2 + x


def main():
    calc_grad()
    # print(calc_speed_cpu())
    # print(calc_speed_gpu())
    # print(optimize(fun=sum_square, x=4.0))  # jnp.array([1.0, 4.0])


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
