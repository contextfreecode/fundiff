import jax
import jax.numpy as jnp
import typing as typ


class Link(typ.NamedTuple):
    angle: float
    length: float


Arm = list[Link]


@jax.jit
def forward(*, arm: Arm) -> jnp.ndarray:
    x = jnp.zeros(2)
    angle = 0.0
    for link in arm:
        angle += link.angle
        x += jnp.array([jnp.cos(angle), jnp.sin(angle)]) * link.length
    return x


def main():
    arm = [
        Link(angle=jnp.pi / 2, length=1.0),
        Link(angle=-jnp.pi / 4, length=1.0),
        Link(angle=-jnp.pi / 4, length=0.5),
    ]
    x = forward(arm=arm)
    print(x)


if __name__ == "__main__":
    main()
