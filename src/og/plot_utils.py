import numpy as np


def padded_minmax(arr, pad_frac: float = 0.02, min_width: float = None):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    arr_min, arr_max = arr.min(), arr.max()
    ptp = arr_max - arr_min
    pad = ptp * pad_frac
    ymin, ymax = arr_min - pad, arr_max + pad

    if min_width is not None:
        if ymax - ymin < min_width:
            mid = (ymax + ymin) / 2
            ymin, ymax = mid - min_width / 2, mid + min_width / 2

    return ymin, ymax
