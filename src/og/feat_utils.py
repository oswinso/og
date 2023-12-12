import jax.numpy as jnp

from og.jax_types import FloatScalar, Vec2, Vec3


def encode_vec2d_exp(vec2: Vec2, eps: float = 1e-5) -> Vec3:
    assert vec2.shape == (2,)
    norm = jnp.sqrt(jnp.sum(vec2**2) + eps)
    vec2_unit = vec2 / norm
    norm_encoded = (jnp.exp(-2 * norm) - 0.15) / 0.15
    return jnp.array([vec2_unit[0], vec2_unit[1], norm_encoded])


def encode_exp(x: FloatScalar, maxval: float) -> FloatScalar:
    out = jnp.exp(-4 * x / maxval)
    return (out - 0.4) / 0.5
