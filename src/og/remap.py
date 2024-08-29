def remap_box(x, lo, hi):
    """Remap x from [-1, 1] to [lo, hi]."""
    return 0.5 * (x + 1) * (hi - lo) + lo
