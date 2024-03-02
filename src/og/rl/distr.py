import jax.numpy as jnp
from jaxtyping import Float

from og.jax_types import Arr


def categorical_l2_project(n_zp: Float[Arr, "n"], n_probs: Float[Arr, "n"], m_zq: Float[Arr, "m"]) -> Float[Arr, "m"]:
    """Projects a categorical distribution (n_zp, n_probs) onto a different support m_zq."""
    (n,) = n_zp.shape
    (m,) = m_zq.shape
    assert n_zp.shape == n_probs.shape == (n,)

    m_dpos = jnp.roll(m_zq, shift=-1)
    m_dneg = jnp.roll(m_zq, shift=1)

    # Clip n_zp to be in the new support range (vmin, vmax)
    n_zp = jnp.clip(n_zp, m_zq[0], m_zq[-1])

    # Get the distance betweeen the atom values in the support.
    m_dpos = m_dpos - m_zq
    m_dneg = m_zq - m_dneg

    # Reshape everything to be be broadcastable to (m, n)
    on_zp, on_probs = n_zp[None, :], n_probs[None, :]
    mo_dpos, mo_dneg, mo_zq = m_dpos[:, None], m_dneg[:, None], m_zq[:, None]

    # Ensure we don't divide by zero when atoms have identical value.
    mo_dpos = jnp.where(mo_dpos > 0, 1.0 / mo_dpos, 0.0)
    mo_dneg = jnp.where(mo_dneg > 0, 1.0 / mo_dneg, 0.0)

    #    clip(n_zp)[j] - m_zq[i]
    mn_delta_qp = on_zp - mo_zq
    mn_d_sign = (mn_delta_qp >= 0.0).astype(n_probs.dtype)

    # Matrix of entries sgn(a_ij) * |a_ij|, where a_ij = clip(n_zp)[j] - m_zq[i]
    mn_delta_hat = (mn_d_sign * mn_delta_qp * mo_dpos) - ((1.0 - mn_d_sign) * mn_delta_qp * mo_dneg)
    m_probs = jnp.sum(jnp.clip(1.0 - mn_delta_hat, 0.0, 1.0) * on_probs, axis=1)
    return m_probs
