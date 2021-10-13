defmodule SimpleMlTest do
  use ExUnit.Case
  doctest SimpleMl

  test "greets the world" do
    assert SimpleMl.hello() == :world
  end
end
