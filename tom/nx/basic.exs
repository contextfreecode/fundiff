defmodule Basic do

  import Nx.Defn

  defn calc_grad do
    x = linspace(-1, 1, 5)
    Nx.transpose(Nx.stack([x, x |> square(), x |> square_grad()]))
  end

  defn calc_speed do
    x = linspace(-1, 1, 100_000_000)
    x |> square() |> Nx.mean()
  end

  @defn_compiler EXLA
  defn calc_speed_cpu do
    calc_speed()
  end

  @defn_compiler {EXLA, client: :cuda}
  defn calc_speed_gpu do
    calc_speed()
  end

  defn linspace(min, max, count) do
    Nx.iota({count}) * (max - min) / (count - 1) + min
  end

  defn square(x) do
    Nx.power(x, 2.0)
  end

  defn square_grad(x) do
    grad(x, fn x -> square(x) end)
  end

  def main do
    IO.inspect(calc_grad())
    # IO.puts("go")
    # IO.inspect(calc_speed_cpu())
    # IO.inspect(calc_speed_gpu())
  end

end

Basic.main()
