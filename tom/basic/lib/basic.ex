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

  # @defn_compiler {EXLA, client: :host}
  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end
end
