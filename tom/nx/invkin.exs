defmodule InvKin do

  import Nx.Defn

  # See https://github.com/elixir-nx/nx/issues/485#issuecomment-926720809

  defn cumsum(x) do
    {n} = Nx.shape(x)
    lower_triangular(n) |> Nx.dot(x)
  end

  defn lower_triangular(n) do
    Nx.iota({n, n}, axis: 1) |> Nx.less_equal(Nx.iota({n, n}, axis: 0))
  end

  defn forward(lengths, angles) do
    angles = cumsum(angles)
    offsets = Nx.stack([Nx.cos(angles), Nx.sin(angles)]) * lengths
    offsets |> Nx.sum(axes: [1])
  end

  defn goal_step(lengths, angles, goal) do
    grad(angles, &loss(lengths, &1, goal))
  end

  def invert(lengths, angles, goal) do
    optimize(&goal_step(lengths, &1, goal), angles)
  end

  defn loss(lengths, angles, goal) do
    Nx.power(forward(lengths, angles) - goal, 2.0) |> Nx.mean()
  end

  def main do
    pi = :math.pi
    angles = Nx.tensor([0.5 * pi, -0.25 * pi, -0.25 * pi])
    lengths = Nx.tensor([1.0, 1.0, 0.5])
    show(lengths)
    goal = Nx.tensor([0.0, 0.5])
    result = invert(lengths, angles, goal)
    show(result)
  end

  def optimize(step, x) do
    rate = 0.2
    nsteps = 20
    for _ <- 1..nsteps, reduce: x do
      x ->
        show(x)
        Nx.subtract(x, Nx.multiply(rate, step.(x)))
    end
  end

  def show(x) do
    IO.inspect(x |> Nx.to_flat_list())
  end

end

InvKin.main()
