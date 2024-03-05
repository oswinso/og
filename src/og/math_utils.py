from og.jax_types import AnyFloat

def rescale_negpos(x: AnyFloat, lo: AnyFloat, hi: AnyFloat):
    """Rescales x from [-1, 1] to [lo, hi]."""
    return (x + 1) * (hi - lo) / 2 + lo


def rescale_01(x: AnyFloat, lo: AnyFloat, hi: AnyFloat) -> AnyFloat:
    """Rescales x from [0, 1] to [lo, hi]."""
    return x * (hi - lo) + lo
