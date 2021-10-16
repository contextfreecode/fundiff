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


Array = typ.Union[float, jnp.ndarray]


def armify(angles: jnp.ndarray, lengths: jnp.ndarray) -> Arm:
    return [
        Link(angle=angle, length=length)
        for angle, length in zip(angles, lengths)
    ]


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


def invert(*, angles: jnp.ndarray, goal: jnp.ndarray, lengths: jnp.ndarray):
    print("invert")
    print(lengths)
    print(angles)
    loss = lambda angles: (
        (forward2(angles=angles, lengths=lengths) - goal) ** 2
    ).mean()
    optimize(fun=loss, x=angles)


def main():
    angles = jnp.array([0.5, -0.25, -0.25]) * jnp.pi
    lengths = jnp.array([1.0, 1.0, 0.5])
    print(forward0(arm=armify(angles=angles, lengths=lengths)))
    # forward_angles = lambda angles: forward2(angles=angles, lengths=lengths)
    # process(angles=angles, forward=forward_angles)
    # process_variety(angles=angles, lengths=lengths)
    # invert(angles=angles, goal=jnp.array([0.0, 0.5]), lengths=lengths)


def optimize(*, fun: typ.Callable[[Array], float], x: Array) -> Array:
    fun_grad = jax.grad(fun)
    rate = 0.2
    nsteps = 20
    for _ in range(nsteps):
        x -= rate * fun_grad(x)
        print(x)
    return x


def process(*, angles: jnp.ndarray, forward: typ.Callable):
    print(forward(angles))
    arm_jac = jax.jacobian(forward)
    jac = arm_jac(angles)
    print(jac)
    print(np.linalg.pinv(jac))


def process_variety(*, angles: jnp.ndarray, lengths: jnp.ndarray):
    forward_angles_funs = [
        lambda angles: forward0(arm=armify(angles=angles, lengths=lengths)),
        lambda angles: forward1(angles=angles, lengths=lengths),
        lambda angles: forward2(angles=angles, lengths=lengths),
    ]
    for forward_angles in forward_angles_funs:
        process(angles=angles, forward=forward_angles)


if __name__ == "__main__":
    main()
