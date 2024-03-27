import functools as ft

import jax.numpy as jnp
import optax
from flax import traverse_util
from loguru import logger

from og.jax_types import FloatScalar


def wd_mask(params):
    Path = tuple[str, ...]
    flat_params: dict[Path, jnp.ndarray] = traverse_util.flatten_dict(params)
    # Apply weight decay to all parameters except biases and LayerNorm scale and bias.
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


def optim(learning_rate: float, wd: float, eps: float, hide_nans: bool):
    opt = optax.adamw(learning_rate, eps=eps, weight_decay=wd, mask=wd_mask)
    if hide_nans:
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
            logger.info("Using apply_if_finite in optimizer to hide NaNs!")
            opt = optax.apply_if_finite(opt, 500)
        return opt

    return optax.inject_hyperparams(ft.partial(optim_, hide_nans=hide_nans))(learning_rate=lr, wd=wd, eps=eps)
