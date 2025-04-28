import functools as ft

import ipdb
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import traverse_util
from jax._src.tree_util import GetAttrKey, keystr
from loguru import logger

from og.jax_types import FloatScalar

_log_apply_if_finite = True


def silence_apply_if_finite():
    global _log_apply_if_finite
    logger.info("Silencing apply_if_finite messages!")
    _log_apply_if_finite = False


def wd_mask(params):
    def f(key_path, val):
        key_path_str = keystr(key_path)
        no_decay = ("bias" in key_path_str) or ("LayerNorm.scale" in key_path_str)
        return not no_decay

    return jtu.tree_map_with_path(f, params)


def optim(learning_rate: float, wd: float, eps: float, hide_nans: bool):
    opt = optax.adamw(learning_rate, eps=eps, weight_decay=wd, mask=wd_mask)
    if hide_nans:
        if _log_apply_if_finite:
            logger.info("Using apply_if_finite in optimizer to hide NaNs!")
        opt = optax.apply_if_finite(opt, 500)
    return opt


def get_default_tx(
    lr: optax.Schedule | FloatScalar,
    wd: optax.Schedule | FloatScalar = 1e-4,
    eps: FloatScalar = 1e-5,
    b1: float = 0.9,
    b2: float = 0.999,
    hide_nans: bool = True,
) -> optax.GradientTransformation:
    def optim_(learning_rate: float, wd: float, eps: float, hide_nans: bool):
        opt = optax.adamw(learning_rate, b1=b1, b2=b2, eps=eps, weight_decay=wd, mask=wd_mask)
        if hide_nans:
            if _log_apply_if_finite:
                logger.info("Using apply_if_finite in optimizer to hide NaNs!")
            opt = optax.apply_if_finite(opt, 500)
        return opt

    return optax.inject_hyperparams(ft.partial(optim_, hide_nans=hide_nans))(learning_rate=lr, wd=wd, eps=eps)
