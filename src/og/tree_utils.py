from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jumpy.numpy as jp
import numpy as np
from jax import tree_util as jtu
from jumpy.core import which_np

from og.jax_types import Arr, Shape
from og.jax_utils import _PyTree, concat_at_front

_PyTree = TypeVar("_PyTree")


def tree_mac(accum: _PyTree, scalar: float, other: _PyTree, strict: bool = True) -> _PyTree:
    """Tree multiply and accumulate. Return accum + scalar * other, but for pytree.
    :rtype: object
    """

    def mac_inner(a, o):
        if strict:
            assert a.shape == o.shape
        return a + scalar * o

    return jtu.tree_map(mac_inner, accum, other)


def tree_add(t1: _PyTree, t2: _PyTree):
    return jtu.tree_map(lambda a, b: a + b, t1, t2)


def tree_inner_product(coefs: list[float], trees: list[_PyTree]) -> _PyTree:
    def tree_inner_product_(*arrs_):
        arrs_ = list(arrs_)
        out = sum([c * a for c, a in zip(coefs, arrs_)])
        return out

    assert len(coefs) == len(trees)
    return jtu.tree_map(tree_inner_product_, *trees)


def tree_split_dims(tree: _PyTree, new_dims: Shape) -> _PyTree:
    prod_dims = np.prod(new_dims)

    def tree_split_dims_inner(arr: Arr) -> Arr:
        assert arr.shape[0] == prod_dims
        target_shape = new_dims + arr.shape[1:]
        return arr.reshape(target_shape)

    return jtu.tree_map(tree_split_dims_inner, tree)


def tree_concat_at_front(tree1: _PyTree, tree2: _PyTree) -> _PyTree:
    return jtu.tree_map(concat_at_front, tree1, tree2)


def tree_stack(trees: list[_PyTree], axis: int = 0) -> _PyTree:
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        return jp.stack(arrs, axis=axis)

    return jtu.tree_map(tree_stack_inner, *trees)


def tree_unstack(tree: _PyTree, axis: int = 0) -> list[_PyTree]:
    """Opposite of tree_stack. Turns a tree of arrays into a list of trees."""

    def get_idx(ii: int):
        return jtu.tree_map(lambda arr: arr[ii], tree)

    return [get_idx(ii) for ii in range(tree_shape(tree, axis=axis))]


def tree_cat(trees: list[_PyTree], axis: int = 0) -> _PyTree:
    def tree_cat_inner(*arrs):
        arrs = list(arrs)
        return jp.concatenate(arrs, axis=axis)

    return jtu.tree_map(tree_cat_inner, *trees)


def tree_split(tree: _PyTree, sections: int, axis: int = 0) -> list[_PyTree]:
    def get_idx(ii: int):
        def tree_split_inner(arr):
            return np.split(arr, sections, axis=axis)[ii]

        return jtu.tree_map(tree_split_inner, tree)

    return [get_idx(ii) for ii in range(sections)]


def tree_where(cond, x_tree: _PyTree, y_tree: _PyTree) -> _PyTree:
    def tree_where_inner(x, y):
        return jp.where(cond, x, y)

    return jax.tree_map(tree_where_inner, x_tree, y_tree)


def tree_where_dim0(cond, x_tree: _PyTree, y_tree: _PyTree) -> _PyTree:
    def tree_where_inner(x, y):
        # x: (b, ...)
        # y: (b, ...)
        # cond: (b, )

        # Get the full shape by broadcasting x and y.
        full_shape = np.broadcast_shapes(x.shape, y.shape)

        cond_reshaped = jp.reshape(cond, (cond.shape[0],) + (1,) * (len(full_shape) - 1))
        return jp.where(cond_reshaped, x, y)

    return jax.tree_map(tree_where_inner, x_tree, y_tree)


def tree_copy(tree: _PyTree) -> _PyTree:
    return jax.tree_map(lambda x: x.copy(), tree)


def tree_len(tree: _PyTree) -> int:
    return tree_shape(tree, axis=0)


def tree_shape(tree: _PyTree, axis: int) -> int:
    leaves, treedef = jtu.tree_flatten(tree)
    return leaves[0].shape[axis]


def tree_unduplicate(tree: _PyTree) -> _PyTree:
    """If there are two arrays that share the same buffer, make them separate."""
    leaves, treedef = jtu.tree_flatten_with_path(tree)


def tree_has_nan(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    for leaf in leaves:
        if jp.any(which_np(leaf).isnan(leaf)):
            return True
    return False


def tree_index(index: int, tree: _PyTree) -> _PyTree:
    return jtu.tree_map(lambda x: x[index], tree)


def make_batch_pytree(tree: _PyTree, size: int, fill_value: int | float = 0, whichnp=None) -> _PyTree:
    """Append a batch dimension to all arrays in the pytree. If it is an int / float, turn it into an array."""

    if whichnp is None:
        whichnp = which_np(tree)

    def fn(val: np.ndarray | jnp.ndarray | float | int | bool):
        if isinstance(val, (np.ndarray, jnp.ndarray)):
            return whichnp.full((size,) + val.shape, fill_value, dtype=val.dtype)
        else:
            # Either float, int, or bool
            dtype = np.float32 if isinstance(val, float) else np.int32 if isinstance(val, int) else bool
            return whichnp.full(size, fill_value, dtype=dtype)

    return jtu.tree_map(fn, tree)
