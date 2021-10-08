defmodule Blah do

  import Nx.Defn

  defn calc_grad do
    x = linspace(-4, 4, 17)
    Nx.transpose(Nx.stack([x, x |> Basic.quad, x |> Basic.quad_grad]))
  end

  defn linspace(min, max, count) do
    Nx.iota({count}) * (max - min) / (count - 1) + min
  end

end

IO.inspect Blah.calc_grad
