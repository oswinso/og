import numpy as np

from og.jax_types import Vec2


def sdf_box_np(pos: Vec2, len_x: float, len_y: float):
    b = np.array([len_x, len_y])
    d = np.abs(pos) - b
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0)
