import functools as ft

import ipdb
import jax
import jax.numpy as jnp
import numpy as np

from og.grad_utils import value_and_jacfwd, value_and_jacrev
from og.jax_utils import jax2np


def test_value_and_jac():
    rng = np.random.default_rng(seed=12345)

    n_degree = 4

    for _ in range(64):
        p = rng.uniform(0.0, 1.0, n_degree + 1)

        def fn(x):
            out1 = jnp.polyval(p, x)
            out2 = jnp.polyval(0.99 * p + 0.01, x)
            return jnp.array([out1, out2])

        x = rng.uniform(-1.0, 1.0, 1)

        val = np.array(fn(x))
        jac_fwd = np.array(jax.jacfwd(fn)(x))
        jac_rev = np.array(jax.jacrev(fn)(x))

        val1, jac1 = jax2np(value_and_jacfwd(fn)(x))
        val2, jac2 = jax2np(value_and_jacrev(fn)(x))

        np.testing.assert_allclose(val, val1)
        np.testing.assert_allclose(val, val2)

        np.testing.assert_allclose(jac_fwd, jac1)
        np.testing.assert_allclose(jac_rev, jac2)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test_value_and_jac()
