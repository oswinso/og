from typing import Any

import equinox as eqx
import jax
import jax._src.traceback_util as traceback_util
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import filter_custom_vjp, tree_pformat

# traceback_util.register_exclusion(__file__)


def backward_nan(x, name=None, terminate=True):
    return _backward_nan(x, name, terminate)


@filter_custom_vjp
def _backward_nan(x, name, terminate):
    return x


@_backward_nan.def_fwd
def _backward_nan_fwd(perturbed, x, name, terminate):
    del perturbed
    return backward_nan(x, name, terminate), None


class _ShortRepr(eqx.Module):
    obj: Any

    def __repr__(self):
        return tree_pformat(self.obj, short_arrays=True)


class _LongRepr(eqx.Module):
    obj: Any

    def __repr__(self):
        return tree_pformat(self.obj, short_arrays=False)


@_backward_nan.def_bwd
def _backward_nan_bwd(residuals, grad_x, perturbed, x, name, terminate):
    del residuals, perturbed
    msg = "   primals ({x_short})\nprimals={x}\ncotangents ({grad_x_short})\ncotangents={grad_x}"
    if name is not None:
        msg = f"{name}:\n" + msg
    jax.debug.print(
        msg,
        x_short=_ShortRepr(x),
        x=_LongRepr(x),
        grad_x_short=_ShortRepr(grad_x),
        grad_x=_LongRepr(grad_x),
        ordered=True,
    )  # pyright: ignore
    if terminate:
        nans = [~jnp.isfinite(a).all() for a in jtu.tree_leaves(eqx.filter(grad_x, eqx.is_array_like))]
        grad_x = eqx.error_if(grad_x, jnp.any(jnp.stack(nans)), f"Encountered NaN in {name}")
    return grad_x
