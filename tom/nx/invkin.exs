defmodule InvKin do

  import Nx.Defn

  # def forward2(*, angles: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
  #   angles = jnp.cumsum(angles)
  #   offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)]) * lengths
  #   return offsets.sum(axis=1)

  defn forward(lengths, angles) do
    Nx.reduce(
      Nx.iota(lengths.shape),
      Nx.tensor([0.0, 0.0]),
      fn i, x ->
        x + Nx.tensor([Nx.cos(angles[i]), Nx.sin(angles[i])]) * lengths[i]
      end
    )
    angles
  end

  defn invert(lengths, angles, goal) do
    forward(lengths, angles)
    Nx.tensor(1)
  end

  def main do
    pi = :math.pi
    angles = Nx.tensor([0.5 * pi, -0.25 * pi, -0.25 * pi])
    lengths = Nx.tensor([1.0, 1.0, 0.5])
    goal = Nx.tensor([0.0, 1.0])
    invert(lengths, angles, goal)
  end

end

InvKin.main()
