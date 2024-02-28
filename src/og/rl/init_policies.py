import numpy as np

from og.tfp import tfd


def uniform_policy(nu: int):
    def fn(*args, **kwargs) -> tfd.Distribution:
        u_lo = np.full(nu, -1.0)
        u_hi = np.full(nu, +1.0)
        return tfd.Independent(tfd.Uniform(low=u_lo, high=u_hi), reinterpreted_batch_ndims=1)

    return fn
