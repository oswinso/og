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

from og.jax_types import Arr, BFloat, BoolScalar, FloatScalar, Shape
