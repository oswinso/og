from typing import TypeVar

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu

from og.jax_types import FloatScalar

_PyTree = TypeVar("_PyTree")


def compute_norm(grad: _PyTree) -> _PyTree:
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad: _PyTree, max_norm: float) -> tuple[_PyTree, FloatScalar]:
    g_norm = compute_norm(grad)
    trigger = jnp.squeeze(g_norm < max_norm)
    assert trigger.shape == tuple()

    def clip_fn(t):
        return lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

    clipped_grad = jtu.tree_map(clip_fn, grad)
    return clipped_grad, g_norm


def compute_norm_and_clip2(grad: _PyTree, max_norm: float) -> tuple[_PyTree, FloatScalar]:
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm
