import dataclasses as dc
import jax
import jax.numpy as jnp
import numpy as np
import typing as typ


# @dc.dataclass
class Link(typ.NamedTuple):
    angle: float
    length: float


Arm = list[Link]


def armify(angles: jnp.ndarray, lengths: jnp.ndarray) -> Arm:
    return [Link(angle=angle, length=length) for angle, length in zip(angles, lengths)]


@jax.jit
def forward0(*, arm: Arm) -> jnp.ndarray:
    x = jnp.zeros(2)
    angle = 0.0
    for link in arm:
        angle += link.angle
        x += jnp.array([jnp.cos(angle), jnp.sin(angle)]) * link.length
    return x


@jax.jit
def forward1(*, angles: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    x = jnp.zeros(2)
    angle = 0.0
    for link_angle, link_length in zip(angles, lengths):
        angle += link_angle
        x += jnp.array([jnp.cos(angle), jnp.sin(angle)]) * link_length
    return x


@jax.jit
def forward2(*, angles: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    angles = jnp.cumsum(angles)
    offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)]) * lengths
    return offsets.sum(axis=1)


def main():
    angles = jnp.array([0.5, -0.25, -0.25]) * jnp.pi
    lengths = jnp.array([1.0, 1.0, 0.5])
    forward_angles_funs = [
        lambda angles: forward0(arm=armify(angles=angles, lengths=lengths)),
        lambda angles: forward1(angles=angles, lengths=lengths),
        lambda angles: forward2(angles=angles, lengths=lengths),
    ]
    for forward_angles in forward_angles_funs:
        process(angles=angles, forward=forward_angles)


def process(*, angles: jnp.ndarray, forward: typ.Callable):
    print(forward(angles))
    arm_jac = jax.jacobian(forward)
    jac = arm_jac(angles)
    print(jac)
    print(np.linalg.pinv(jac))


if __name__ == "__main__":
    main()
