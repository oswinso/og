from typing import Callable, Sequence

import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, din: int, hid_sizes: Sequence[int], *, activation: Callable, rngs: nnx.Rngs):
        self.linears = []
        for hid_size in hid_sizes:
            self.linears.append(nnx.Linear(din, hid_size, rngs=rngs))
            din = hid_size
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
        return x


class LayerNormMLP(nnx.Module):
    def __init__(
        self,
        din: int,
        hid_sizes: Sequence[int],
        *,
        activation: Callable,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        use_offset: bool = True,
        use_scale: bool = True
    ):
        self.linears = []
        self.layernorms = []
        for hid_size in hid_sizes:
            self.linears.append(nnx.Linear(din, hid_size, use_bias=use_bias, rngs=rngs))
            self.layernorms.append(nnx.LayerNorm(hid_size, use_bias=use_offset, use_scale=use_scale, rngs=rngs))
            din = hid_size
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for linear, layernorm in zip(self.linears, self.layernorms):
            x = linear(x)
            x = layernorm(x)
            x = self.activation(x)
        return x
