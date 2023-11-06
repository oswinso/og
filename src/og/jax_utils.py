import functools as ft
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar

import einops as ei
import ipdb
import jax._src.dtypes
import jax.config
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.lib import xla_client as xc
from jax._src.typing import ArrayLike
from loguru import logger

from og.jax_types import Arr, BFloat, BoolScalar, FloatScalar

_PyTree = TypeVar("_PyTree")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]


def jax_use_double():
    jax.config.update("jax_enable_x64", True)


def jax_default_x32():
    jax.config.update("jax_default_dtype_bits", "32")
    reset_default_types()


def reset_default_types():
    is_32 = jax.config.jax_default_dtype_bits == "32"
    jax._src.dtypes.int_ = np.int32 if is_32 else np.int64
    jax._src.dtypes.uint = np.uint32 if is_32 else np.uint64
    jax._src.dtypes.float_ = np.float32 if is_32 else np.float64
    jax._src.dtypes.complex_ = np.complex64 if is_32 else np.complex128
    jax._src.dtypes._default_types = {
        "b": jax._src.dtypes.bool_,
        "i": jax._src.dtypes.int_,
        "u": jax._src.dtypes.uint,
        "f": jax._src.dtypes.float_,
        "c": jax._src.dtypes.complex_,
    }
    dtypes = jax._src.dtypes

    jax.numpy.int_ = jnp.int32 if dtypes.int_ == np.int32 else jnp.int64
    jax.numpy.uint = jnp.uint32 if dtypes.uint == np.uint32 else jnp.uint64
    jax.numpy.float_ = jnp.float32 if dtypes.float_ == np.float32 else jnp.float64
    jax.numpy.complex_ = jnp.complex64 if dtypes.complex_ == np.complex64 else jnp.complex128


def get_cpu_device(idx: int = 0):
    return jax.devices("cpu")[idx]


def get_gpu_device(idx: int = 0):
    return jax.devices("gpu")[idx]


def jax_use_cpu() -> None:
    ctx = jax.default_device(get_cpu_device())
    ctx.__enter__()


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)
