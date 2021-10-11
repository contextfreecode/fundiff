import jax
import jax.numpy as jnp
import numpy as np
import typing as typ


class Link(typ.NamedTuple):
    angle: float
    length: float


Arm = list[Link]


def armify(angles: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
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
    print(forward0(arm=armify(angles=angles, lengths=lengths)))
    print(forward1(angles=angles, lengths=lengths))
    print(forward2(angles=angles, lengths=lengths))
    forward0a = lambda angles: forward0(arm=armify(angles=angles, lengths=lengths))
    arm_jac0 = jax.jacfwd(forward0a)
    print(arm_jac0(angles))
    forward1a = lambda angles: forward1(angles=angles, lengths=lengths)
    arm_jac1 = jax.jacfwd(forward1a)
    print(arm_jac1(angles))
    forward2a = lambda angles: forward2(angles=angles, lengths=lengths)
    arm_jac2 = jax.jacfwd(forward2a)
    print(arm_jac2(angles))
    print(np.linalg.pinv(arm_jac2(angles)))
    # Was hanging on jnp.linalg.pinv, now crashing,
    # print(jnp.linalg.pinv(arm_jac2(angles)))


if __name__ == "__main__":
    main()
