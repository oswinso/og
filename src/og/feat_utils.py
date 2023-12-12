import jax.numpy as jnp
import numpy as np

from og.jax_types import FloatScalar, Vec2, Vec3


def encode_vec2d_exp(vec2: Vec2, eps: float = 1e-5) -> Vec3:
    assert vec2.shape == (2,)
    norm = jnp.sqrt(jnp.sum(vec2**2) + eps)
    vec2_unit = vec2 / norm
    norm_encoded = (jnp.exp(-2 * norm) - 0.15) / 0.15
    return jnp.array([vec2_unit[0], vec2_unit[1], norm_encoded])


def encode_vec2d_log(vec2: Vec2, max_dist: float, eps: float = 1e-5) -> Vec3:
    assert vec2.shape == (2,)
    norm = jnp.sqrt(jnp.sum(vec2**2) + eps)
    vec2_unit = vec2 / norm
    norm_encoded = jnp.log(1 + norm)
    # Normalize norm_encoded to be [0, 1] within [0, max_dist].
    norm_encoded = norm_encoded / np.log(1 + max_dist)
    # Normalize norm_encoded to be [-1, 1].
    norm_encoded = norm_encoded * 2 - 1
    # Clip it to prevent OOD.
    norm_encoded = norm_encoded.clip(min=-1, max=1)
    return jnp.array([vec2_unit[0], vec2_unit[1], norm_encoded])


def encode_exp(x: FloatScalar, maxval: float) -> FloatScalar:
    out = jnp.exp(-4 * x / maxval)
    return (out - 0.4) / 0.5
