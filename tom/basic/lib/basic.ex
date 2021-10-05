defmodule Basic do
  import Nx.Defn
    @moduledoc """
  Documentation for `Basic`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Basic.hello()
      :world

  """
  def hello do
    :world
  end

  # See:
  # https://github.com/elixir-nx/nx/blob/main/exla/bench/softmax.exs
  # https://github.com/elixir-nx/nx/tree/main/nx#readme

  # @defn_compiler {EXLA, client: :host}
  defn softmax(x) do
    Nx.exp(x) / Nx.sum(Nx.exp(x))
  end

  @defn_compiler EXLA
  defn host(x), do: softmax(x)

  @defn_compiler {EXLA, client: :cuda}
  defn cuda(x), do: softmax(x)

  @defn_compiler {EXLA, client: :cuda, run_options: [keep_on_device: true]}
  defn cuda_keep(x), do: softmax(x)
end
