defmodule Basic do

  import Nx.Defn

  defn calc_grad do
    x = linspace(-4, 4, 17)
    Nx.transpose(Nx.stack([x, x |> quad(), x |> quad_grad()]))
  end

  @defn_compiler {EXLA, client: :cuda}
  # @defn_compiler EXLA
  defn calc_speed do
    x = linspace(-1, 1, 100_000_000)
    Nx.stack([
      x |> quad() |> Nx.size() |> Nx.tensor(),
      x |> quad_grad() |> Nx.size() |> Nx.tensor(),
    ])
  end

  defn linspace(min, max, count) do
    Nx.iota({count}) * (max - min) / (count - 1) + min
  end

  defn quad(x) do
    Nx.power(x, 2.0)
  end

  defn quad_grad(x) do
    grad(x, fn x -> quad(x) end)
  end

end

IO.inspect Basic.calc_grad
IO.inspect Basic.calc_speed
