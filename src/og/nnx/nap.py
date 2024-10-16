import jax.numpy as jnp
from flax import nnx

from og.jax_types import FloatScalar
from og.nnx.mlp import LayerNormMLP


class NAPProjector(nnx.Module):
    def __init__(self, nn: nnx.Module):
        # Compute the initial norms of all the weights for all the LayerNormMLPs.
        self.initial_norms = {}
        for path, m in nn.iter_modules():
            if isinstance(m, LayerNormMLP):
                initial_norms = [jnp.sqrt(jnp.sum(linear.kernel**2)) for linear in m.linears]
                self.initial_norms[path] = initial_norms

    def __call__(self, nn: nnx.Module):
        # Project all LayerNormMLP in the modules.
        for path, m in nn.iter_modules():
            if isinstance(m, LayerNormMLP):
                assert path in self.initial_norms
                nap_project(m, self.initial_norms[path])


def nap_project(nn: LayerNormMLP, initial_norms: list[FloatScalar]):
    assert len(initial_norms) == len(nn.linears)
    eps = 0.0

    # Scale the weights so that the initial norms are preserved.
    for initial_norm, linear in zip(initial_norms, nn.linears):
        current_norm = jnp.sum(linear.kernel**2)
        scale = initial_norm / jnp.sqrt(eps + current_norm)
        linear.kernel.value = linear.kernel.value * scale

    # Scale the scale and offset
    for layernorm in nn.layernorms:
        assert layernorm.use_bias and layernorm.use_scale

        norm = jnp.sum(layernorm.scale**2 + layernorm.bias**2)
        d = layernorm.num_features
        scale = jnp.sqrt(eps + d / norm)
        layernorm.scale.value = layernorm.scale.value * scale
        layernorm.bias.value = layernorm.bias.value * scale
