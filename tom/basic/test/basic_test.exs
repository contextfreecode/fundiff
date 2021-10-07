defmodule BasicTest do
  use ExUnit.Case
  doctest Basic

  test "greets the world" do
    assert Basic.hello() == :world
  end

  test "nx" do
    x = Nx.tensor([[1, 2], [3, 4]])
    Basic.softmax(x)
  end

  test "grad" do
    assert Basic.quad_grad(1.0) == Nx.tensor(2.0)
  end
end
