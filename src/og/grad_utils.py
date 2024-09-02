from typing import ParamSpec, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.api import *
from jax._src.api import (
    _check_input_dtype_jacfwd,
    _check_input_dtype_jacrev,
    _check_output_dtype_jacfwd,
    _check_output_dtype_jacrev,
    _jacfwd_unravel,
    _jacrev_unravel,
    _jvp,
    _std_basis,
    _vjp,
)
from jax._src.api_util import (
    _ensure_index,
    argnums_partial,
    check_callable,
)

from og.jax_types import FloatScalar

_PyTree = TypeVar("_PyTree")

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]


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


# def value_and_jacfwd(f: _Fn, has_aux: bool = False):
#     def value_and_jacfwd_f(x):
#         basis = jnp.eye(x.size, dtype=x.dtype)
#
#         if not has_aux:
#             pushfwd: Callable = ft.partial(jax.jvp, f, (x,))
#             y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis,))
#             return y, jac
#         else:
#             pushfwd: Callable = ft.partial(jax.jvp, f, (x,), has_aux=True)
#             y, jac, aux = jax.vmap(pushfwd, out_axes=(None, -1, None))((basis,))
#             return y, jac, aux
#
#     return value_and_jacfwd_f
#
#
# def value_and_jacrev(f: _Fn, has_aux: bool = False):
#     def value_and_jacrev_f(x):
#         if not has_aux:
#             y, pullback = jax.vjp(f, x)
#             basis = jnp.eye(y.size, dtype=x.dtype)
#             jac = jax.vmap(pullback)(basis)
#             return y, jac
#         else:
#             y, pullback, aux = jax.vjp(f, x, has_aux=True)
#             basis = jnp.eye(y.size, dtype=x.dtype)
#             jac = jax.vmap(pullback)(basis)
#             return y, jac, aux
#
#     return value_and_jacrev_f
#
#


def value_and_jacfwd(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, holomorphic: bool = False, has_aux: bool = False
) -> Callable:
    """
    [Constructed by analogy to value_and_grad -- see help(value_and_grad) for more.]
    """
    check_callable(fun)
    argnums = _ensure_index(argnums)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if has_aux:
            return y, aux, jac_tree
        else:
            return y, jac_tree

    return jacfun


def value_and_jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    has_aux: bool = False,
    allow_int: bool = False,
) -> Callable:
    """
    [Constructed by analogy to value_and_grad -- see help(value_and_grad) for more.]
    """
    check_callable(fun)
    argnums = _ensure_index(argnums)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)
        if has_aux:
            return y, aux, jac_tree
        else:
            return y, jac_tree

    return jacfun


jax.value_and_jacfwd = value_and_jacfwd  # !!
