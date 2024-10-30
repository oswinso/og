from typing import Sequence

import numpy as np


def cat_last(arrs: Sequence[np.ndarray]):
    """
    Given a list of arrays with either shape (T, ) or (T, n), concatenate them along the last axis, resulting in
    shape (T, sum(n)).
    """
    # 1: Make all arrays be either (T, 1) or (T, n).
    arrs_2d = []
    for arr in arrs:
        assert arr.ndim in [1, 2]
        if arr.ndim == 1:
            arr = arr[:, None]
        arrs_2d.append(arr)

    # 2: Concatenate.
    return np.concatenate(arrs_2d, axis=-1)
