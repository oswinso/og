from typing import Sequence

import equinox as eqx
import ipdb
import jax.debug as jd
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from loguru import logger

from og.jax_types import Arr, FloatScalar
from og.nan_utils import backward_nan


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


def midpoint_to_endpoints(n_midpoints: Float[Arr, "n"]) -> Float[Arr, "n+1"]:
    """Converts a set of midpoints to endpoints."""
    (n,) = n_midpoints.shape
    assert n > 1, "There must be at least two midpoints to determine the bin sizes."
    nm1_endpoints = (n_midpoints[1:] + n_midpoints[:-1]) / 2
    width_first = nm1_endpoints[0] - n_midpoints[0]
    width_last = n_midpoints[-1] - nm1_endpoints[-1]

    endpoint_first = n_midpoints[0] - width_first
    endpoint_last = n_midpoints[-1] + width_last

    np1_endpoints = jnp.concatenate([jnp.array([endpoint_first]), nm1_endpoints, jnp.array([endpoint_last])], axis=0)
    assert np1_endpoints.shape == (n + 1,)
    return np1_endpoints


def sample_cvar_cvxcomb(alpha: FloatScalar, n_zp: Float[Arr, "n"]) -> FloatScalar:
    (n,) = n_zp.shape
    n_probs = jnp.full(n, 1 / n)
    return categorical_cvar_cvxcomb(alpha, n_zp, n_probs)


def categorical_var_cvxcomb(alpha: FloatScalar, n_zp: Float[Arr, "n"], n_probs: Float[Arr, "n"]) -> FloatScalar:
    """Calculate VaR of categorical distribution in the same way as for convex combination CVaR."""
    (n,) = n_zp.shape

    # 1: Compute the alpha-quantile.
    n_probs_cumsum = jnp.cumsum(n_probs)
    #     [ 0.0, 0.0, 0.5, 1.0 ]
    #     left: Returns 0 for (-∞, 0], 2 for (0, 0.5], 3 for (0.5, 1.0], 4 for (1.0, ∞)
    #     right: Returns 0 for (-∞, 0), 2 for [0, 0.5), 3 for [0.5, 1.0), 4 for [1.0, ∞)
    idx = jnp.searchsorted(n_probs_cumsum, alpha, side="right")
    VaR = jnp.array(n_zp)[idx]
    return VaR


def categorical_var_cvxcomb_np(alpha: FloatScalar, n_zp: Float[Arr, "n"], n_probs: Float[Arr, "n"]) -> FloatScalar:
    """Calculate VaR of categorical distribution in the same way as for convex combination CVaR."""
    (n,) = n_zp.shape

    # 1: Compute the alpha-quantile.
    n_probs_cumsum = np.cumsum(n_probs)
    #     [ 0.0, 0.0, 0.5, 1.0 ]
    #     left: Returns 0 for (-∞, 0], 2 for (0, 0.5], 3 for (0.5, 1.0], 4 for (1.0, ∞)
    #     right: Returns 0 for (-∞, 0), 2 for [0, 0.5), 3 for [0.5, 1.0), 4 for [1.0, ∞)
    idx = np.searchsorted(n_probs_cumsum, alpha, side="right")
    VaR = np.array(n_zp)[idx]
    return VaR


def categorical_cvar_cvxcomb(alpha: FloatScalar, n_zp: Float[Arr, "n"], n_probs: Float[Arr, "n"]) -> FloatScalar:
    """Calculate CVaR of categorical distribution using the convex combination formula."""
    (n,) = n_zp.shape

    if alpha == 0.0:
        # Return the mean.
        logger.info("alpha=0, so returning mean!")
        return jnp.dot(n_zp, n_probs)

    # 1: Compute the alpha-quantile.
    n_probs_cumsum = jnp.cumsum(n_probs)

    #     [ 0.0, 0.0, 0.5, 1.0 ]
    #     left: Returns 0 for (-∞, 0], 2 for (0, 0.5], 3 for (0.5, 1.0], 4 for (1.0, ∞)
    #     right: Returns 0 for (-∞, 0), 2 for [0, 0.5), 3 for [0.5, 1.0), 4 for [1.0, ∞)
    idx = jnp.searchsorted(n_probs_cumsum, alpha, side="right")

    idx = eqx.error_if(idx, idx == len(n_zp), "idx == len(n_zp)!")
    idx = eqx.error_if(idx, idx < 0, "idx < 0!")

    VaR = jnp.array(n_zp)[idx]
    cdf_VaR = n_probs_cumsum[idx]

    lambd = (cdf_VaR - alpha) / (1 - alpha)

    # 2: Compute the "strict" CVaR^+.
    n_probs_cond = jnp.where(jnp.arange(n) > idx, n_probs, 0.0)
    sum_n_probs_cond = jnp.sum(n_probs_cond)
    # sum_n_probs_cond = backward_nan(sum_n_probs_cond, "sum_n_probs_cond")

    # jd.print("n_probs_cond: {}", n_probs_cond)
    # jd.print("n_probs_cond_sum: {}", sum_n_probs_cond)

    # IMPORTANT: Double where to avoid NaN in grad.
    # However, if the sum is tiny (e.g., 1e-22), then the reciprocal will be huge (1e22).
    # This leads to NaN / infs for the gradient, since the gradient will involve the square of the reciprocal (1e44)
    recip_eps = 1e-6
    sum_n_probs_cond_safe = jnp.where(sum_n_probs_cond > recip_eps, sum_n_probs_cond, 1.0)
    # sum_n_probs_cond_safe = backward_nan(sum_n_probs_cond_safe, "sum_n_probs_cond")

    cond_expectation_denom_recip = jnp.where(sum_n_probs_cond_safe > recip_eps, 1 / sum_n_probs_cond_safe, 1.0)

    # n_probs_cond = backward_nan(n_probs_cond, "n_probs_cond")
    # cond_expectation_denom_recip = backward_nan(cond_expectation_denom_recip, "cond_expectation_denom_recip")

    n_probs_cond_norm = n_probs_cond * cond_expectation_denom_recip
    # n_probs_cond_norm = backward_nan(n_probs_cond_norm, "n_probs_cond_norm")

    # CVaR_plus = jnp.dot(n_zp, n_probs_cond) * cond_expectation_denom_recip
    CVaR_plus = jnp.dot(n_zp, n_probs_cond_norm)

    # CVaR_plus = backward_nan(CVaR_plus, "CVaR_plus")

    # 3: Compute CVaR using convex combination of VaR and CVaR^+.
    CVaR = lambd * VaR + (1 - lambd) * CVaR_plus
    return CVaR


def categorical_cvar_cvxcomb_logp(alpha: FloatScalar, n_zp: Float[Arr, "n"], n_logp: Float[Arr, "n"]) -> FloatScalar:
    """Calculate CVaR of categorical distribution using the convex combination formula. Take log probs as input."""
    (n,) = n_zp.shape

    # -----------------------------------------------------
    # 1: Compute the alpha-quantile.
    n_probs = jnp.exp(n_logp)
    n_probs_cumsum = jnp.cumsum(n_probs)

    #     [ 0.0, 0.0, 0.5, 1.0 ]
    #     left: Returns 0 for (-∞, 0], 2 for (0, 0.5], 3 for (0.5, 1.0], 4 for (1.0, ∞)
    #     right: Returns 0 for (-∞, 0), 2 for [0, 0.5), 3 for [0.5, 1.0), 4 for [1.0, ∞)
    idx = jnp.searchsorted(n_probs_cumsum, alpha, side="right")

    idx = eqx.error_if(idx, idx == len(n_zp), "idx == len(n_zp)!")
    idx = eqx.error_if(idx, idx < 0, "idx < 0!")

    VaR = jnp.array(n_zp)[idx]
    cdf_VaR = n_probs_cumsum[idx]
    lambd = (cdf_VaR - alpha) / (1 - alpha)

    # -----------------------------------------------------
    # 2: Compute the "strict" CVaR^+.
    #        Compute the conditional probability.
    big_neg_num = -100.0
    n_logp_cond_unnorm = jnp.where(jnp.arange(n) > idx, n_logp, big_neg_num)
    n_logp_cond_norm = n_logp_cond_unnorm - jnn.logsumexp(n_logp_cond_unnorm)
    n_probs_cond = jnp.exp(n_logp_cond_norm)
    CVaR_plus = jnp.dot(n_zp, n_probs_cond)

    # 3: Compute CVaR using convex combination of VaR and CVaR^+.
    CVaR = lambd * VaR + (1 - lambd) * CVaR_plus
    return CVaR


def categorical_cvar_cvxcomb_np(alpha: FloatScalar, n_zp: Float[Arr, "n"], n_probs: Float[Arr, "n"]) -> FloatScalar:
    """Calculate CVaR of categorical distribution using the convex combination formula."""
    (n,) = n_zp.shape

    # 1: Compute the alpha-quantile.
    n_probs_cumsum = np.cumsum(n_probs)
    #     [ 0.0, 0.0, 0.5, 1.0 ]
    #     left: Returns 0 for (-∞, 0], 2 for (0, 0.5], 3 for (0.5, 1.0], 4 for (1.0, ∞)
    #     right: Returns 0 for (-∞, 0), 2 for [0, 0.5), 3 for [0.5, 1.0), 4 for [1.0, ∞)
    idx = np.searchsorted(n_probs_cumsum, alpha, side="right")
    VaR = np.array(n_zp)[idx]
    cdf_VaR = n_probs_cumsum[idx]

    lambd = (cdf_VaR - alpha) / (1 - alpha)

    # 2: Compute the "strict" CVaR^+.
    n_probs_cond = np.where(np.arange(n) > idx, n_probs, 0.0)
    sum_n_probs_cond = np.sum(n_probs_cond)
    cond_expectation_denom_recip = np.where(sum_n_probs_cond > 0, 1 / sum_n_probs_cond, 0.0)
    CVaR_plus = np.dot(n_zp, n_probs_cond) * cond_expectation_denom_recip

    # 3: Compute CVaR using convex combination of VaR and CVaR^+.
    CVaR = lambd * VaR + (1 - lambd) * CVaR_plus
    return CVaR


def categorical_cvar_unif(alpha: FloatScalar, n_zp: Float[Arr, "n"], n_probs: Float[Arr, "n"]) -> FloatScalar:
    """Compute the CVaR of a categorical distribution."""
    (n,) = n_zp.shape

    # 1: Compute the alpha-quantile.
    n_probs_cumsum = jnp.cumsum(n_probs)
    #     [ 0.0, 0.0, 0.5, 1.0 ]
    #     left: Returns 0 for (-∞, 0], 2 for (0, 0.5], 3 for (0.5, 1.0], 4 for (1.0, ∞)
    #     right: Returns 0 for (-∞, 0), 2 for [0, 0.5), 3 for [0.5, 1.0), 4 for [1.0, ∞)
    idx = jnp.searchsorted(n_probs_cumsum, alpha, side="right")

    # If idx = 0, then the probability remaining in this bin is alpha.
    p_prev = jnp.where(idx > 0, n_probs_cumsum[idx - 1], 0.0)

    # 2: Compute the probability for each bin being greater than the alpha-quantile.
    n_cond_probs = jnp.where(jnp.arange(n) > idx, n_probs, 0.0)

    np1_endpoints = jnp.array(midpoint_to_endpoints(n_zp))
    bin_L, bin_R = np1_endpoints[idx], np1_endpoints[idx + 1]

    prob_bin = jnp.array(n_probs)[idx]
    prob_bin_recip = jnp.where(prob_bin > 0, 1.0 / prob_bin, 0.0)
    p_bin_less = alpha - p_prev
    p_bin_greater = prob_bin - p_bin_less
    frac_bin_less = p_bin_less * prob_bin_recip

    VaR = (1 - frac_bin_less) * bin_L + frac_bin_less * bin_R
    bin_mean = (VaR + bin_R) / 2

    # 3: Compute the CVaR.
    cond_mean = jnp.dot(n_zp, n_cond_probs) + p_bin_greater * bin_mean
    CVaR = cond_mean / (1 - alpha)

    return CVaR


def categorical_crossentropy(p: Float[Arr, "n"], q_logit: Float[Arr, "n"], axis: int | None | Sequence[int] = None):
    """Compute the cross-entropy between two categorical distributions."""
    logq = jnn.log_softmax(q_logit)
    ce_out = -jnp.sum(p * logq, axis=axis)
    return ce_out


def categorical_entropy(p: Float[Arr, "n"], axis: int | None | Sequence[int] = None) -> FloatScalar:
    """Compute the entropy of a categorical distribution."""
    logp = jnp.log(p)
    # Prevent 0 * log(0) = nan.
    logp_safe = jnp.where(p > 0, logp, 0.0)
    entropy_out = -jnp.sum(p * logp_safe, axis=axis)
    return entropy_out


def categorical_kl(
    p: Float[Arr, "n"], q_logit: Float[Arr, "n"], axis: int | None | Sequence[int] = None
) -> FloatScalar:
    """Compute KL( P || Q )."""
    logq = jnn.log_softmax(q_logit)
    logp = jnp.log(p)
    # Prevent 0 * log(0) = nan.
    logp_safe = jnp.where(p > 0, logp, 0.0)
    kl_out = jnp.sum(p * (logp_safe - logq), axis=axis)
    kl_out = eqx.error_if(kl_out, kl_out < 0, "KL divergence is negative.")
    return kl_out


def categorical_max(n_prob1: Float[Arr, "n"], n_prob2: Float[Arr, "n"]):
    n, = n_prob1.shape
    assert n_prob1.shape == n_prob2.shape == (n,)
    # Just for notation.
    m_prob2 = n_prob2

    nm_jointprob = jnp.outer(n_prob1, m_prob2)

    # Assume that the values are sorted in ascending order.
    n_val1 = jnp.arange(n)
    m_val2 = jnp.arange(n)
    nm_maxval = jnp.maximum(n_val1[:, None], m_val2[None, :])

    # Add the probabilities of the joint distribution that have the same maximum value. Probably a scatter.
    n_probs = jnp.zeros(n)
    n_probs = n_probs.at[nm_maxval].add(nm_jointprob)

    return n_probs
