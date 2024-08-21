import numpy as np
import jumpy.numpy as jp


def in_triangle(pos: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Solve using barycentric coordinates."""
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = pos - p0

    dot00 = jp.dot(v0, v0)
    dot01 = jp.dot(v0, v1)
    dot02 = jp.dot(v0, v2)
    dot11 = jp.dot(v1, v1)
    dot12 = jp.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) & (v >= 0) & (u + v < 1)
