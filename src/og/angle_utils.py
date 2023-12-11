import jax.numpy as jnp
import numpy as np

from og.jax_types import FloatScalar, RotMat2D


def rot2d(theta: FloatScalar) -> RotMat2D:
    if isinstance(theta, jnp.ndarray):
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[c, -s], [s, c]])
    else:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
