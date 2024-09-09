def remap_box(x, lo, hi):
    """Remap x from [-1, 1] to [lo, hi]."""
    return 0.5 * (x + 1) * (hi - lo) + lo


def remap_box_inv(y, lo, hi):
    """The inverse of remap_box. From [lo, hi] to [-1, 1]"""
    return 2 * (y - lo) / (hi - lo) - 1
