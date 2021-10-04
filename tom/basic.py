import jax
import jax.numpy as jnp


def quad(x):
    return x ** 2


def main():
    quad_grad = jax.grad(lambda x: quad(x).sum())
    x = jnp.linspace(-4, 4, 17)
    y = jnp.vstack([x, quad(x), quad_grad(x)])
    print(y.T)


main()
