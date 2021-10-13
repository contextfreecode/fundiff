defmodule InvKin do

  import Nx.Defn

  # def forward2(*, angles: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
  #   angles = jnp.cumsum(angles)
  #   offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)]) * lengths
  #   return offsets.sum(axis=1)

  defn forward(lengths, angles) do
    # Nx.stack([lengths, angles])
    Nx.reduce(
      Nx.stack([lengths, angles]),
      0.0,
      [axes: [1]],
      fn length_angle, x ->
        # angle = length_angle[1]
        # length = length_angle[0]
        # Nx.add(x, Nx.tensor([Nx.cos(angle), Nx.sin(angle)]) * length)
        Nx.add(length_angle, x)
      end
    )
  end

  defn invert(lengths, angles, goal) do
    # Nx.stack([lengths, angles])
    forward(lengths, angles)
    # angles
  end

  def main do
    pi = :math.pi
    angles = Nx.tensor([0.5 * pi, -0.25 * pi, -0.25 * pi])
    lengths = Nx.tensor([1.0, 1.0, 0.5])
    goal = Nx.tensor([0.0, 1.0])
    x = invert(lengths, angles, goal)
    IO.inspect(x)
  end

end

InvKin.main()
