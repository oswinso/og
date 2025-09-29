import functools as ft
import inspect
from typing import Any, Callable, Concatenate, Generic, NamedTuple, ParamSpec, TypeVar

import flax.linen as nn
import ipdb
import jax.tree_util as jtu
import optax
from flax import struct
from jaxtyping import PyTreeDef
from loguru import logger
from optax._src.wrappers import ApplyIfFiniteState

from og.jax_types import FloatScalar
from og.none import get_or
from og.rng import PRNGKey

_Params = TypeVar("_Params")

_P = ParamSpec("_P")
_R = TypeVar("_R")
_C = TypeVar("_C")
_N = TypeVar("_N", bound=NamedTuple)
_ApplyFn = Callable[Concatenate[_Params, _P], _R]

_OApplyFn = Callable[Concatenate[_Params, _P], _N]


class ParamState(Generic[_P], struct.PyTreeNode):
    """Non-trainable version of TrainState."""

    apply_fn: _ApplyFn = struct.field(pytree_node=False)
    params: _Params


class TrainState(Generic[_R], struct.PyTreeNode):
    """Custom version of flax.training.TrainState but with better type information."""

    step: int
    apply_fn: _ApplyFn = struct.field(pytree_node=False)
    params: _Params
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState | optax.InjectHyperparamsState

    def vars_dict(self, params: _Params | None = None):
        params = get_or(params, self.params)
        return {"params": params}

    def apply_mut(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.apply(*args, **kwargs)

    def apply(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.apply_fn(self.vars_dict(), *args, **kwargs)

    def apply_with(self, *args: _P.args, params: _Params, **kwargs: _P.kwargs) -> _R:
        return self.apply_fn(self.vars_dict(params), *args, **kwargs)

    def apply_gradients(self, grads: _Params, **kwargs) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def set_batch_stats(self, batch_stats: dict) -> "TrainState":
        return self.replace(batch_stats=batch_stats)

    def set_lr(self, lr: FloatScalar):
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                hyperparams[key] = lr
                return
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @property
    def lr(self) -> FloatScalar:
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                return hyperparams[key]
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @property
    def total_notfinite(self) -> int:
        inner_state = self.opt_state.inner_state
        assert isinstance(inner_state, ApplyIfFiniteState)
        return inner_state.total_notfinite

    @classmethod
    def create(
        cls,
        apply_fn: _ApplyFn,
        params: _Params,
        tx: optax.GradientTransformation,
        **kwargs,
    ) -> "TrainState":
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    @classmethod
    def create_from_def(
        cls, key: PRNGKey, net_def: nn.Module, init_args: tuple, tx: optax.GradientTransformation, **kwargs
    ) -> "TrainState":
        variables: dict = net_def.init(key, *init_args)
        params = variables["params"]
        return cls.create(net_def.apply, params, tx, **kwargs)

    def strip(self) -> "TrainState":
        """Remove tx and opt_state."""
        return self.replace(tx=None, opt_state=None)


class EqTrainState(Generic[_C], struct.PyTreeNode):
    step: int
    params: list[Any]
    model_treedef: PyTreeDef = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState | optax.InjectHyperparamsState

    @property
    def model(self) -> _C:
        return jtu.tree_unflatten(self.model_treedef, self.params)

    def model_with(self, params: _Params) -> _C:
        return jtu.tree_unflatten(self.model_treedef, params)

    def apply_gradients(self, grads: _Params, **kwargs) -> "EqTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def set_lr(self, lr: FloatScalar):
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                hyperparams[key] = lr
                return
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @property
    def lr(self) -> FloatScalar:
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                return hyperparams[key]
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @classmethod
    def create(
        cls,
        model: _C,
        tx: optax.GradientTransformation,
        **kwargs,
    ) -> "EqTrainState":
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        flat_model, treedef_model = jtu.tree_flatten(model)
        del model

        opt_state = None
        if tx is not None:
            opt_state = tx.init(flat_model)
        return cls(
            step=0,
            params=flat_model,
            model_treedef=treedef_model,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class BNTrainState(TrainState[_R]):
    """To avoid breaking everything, make a subclass."""

    batch_stats: dict

    def vars_dict(self, params: _Params | None = None):
        params = get_or(params, self.params)
        batch_stats_dict = {}
        if len(self.batch_stats) > 0:
            batch_stats_dict = {"batch_stats": self.batch_stats}
        return {"params": params} | batch_stats_dict

    def mut_dict(self):
        mutable_dict = {}
        if len(self.batch_stats) > 0:
            mutable_dict = dict(mutable=["batch_stats"])
        return mutable_dict

    def apply_mut(self, *args: _P.args, **kwargs: _P.kwargs) -> tuple[_R, dict]:
        return self.apply(*args, **self.mut_dict(), **kwargs)

    def apply_with_run_avg(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        use_run_avg = True
        return self.apply_fn(self.vars_dict(), *args, use_run_avg, **kwargs)

    def apply_mut_with(self, *args: _P.args, params: _Params, **kwargs: _P.kwargs) -> tuple[_R, dict]:
        return self.apply_fn(self.vars_dict(params), *args, **self.mut_dict(), **kwargs)

    @classmethod
    def create(
        cls,
        apply_fn: _ApplyFn,
        params: _Params,
        tx: optax.GradientTransformation,
        batch_stats: dict | None = None,
        **kwargs,
    ) -> "TrainState":
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)
        batch_stats = get_or(batch_stats, {})
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    @classmethod
    def create_from_def(
        cls, key: PRNGKey, net_def: nn.Module, init_args: tuple, tx: optax.GradientTransformation, **kwargs
    ) -> "TrainState":
        variables: dict = net_def.init(key, *init_args)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", {})
        return cls.create(net_def.apply, params, tx, batch_stats=batch_stats, **kwargs)


class NT(NamedTuple):
    a: nn.Dense
    b: nn.Dense


class Modules(nn.Module, Generic[_N]):
    modules: _N

    @nn.compact
    def __call__(self, *args, name=None, args_dict: NamedTuple | None = None, **kwargs):
        if name is None:
            out = {}
            for name in self.modules._fields:
                module = getattr(self.modules, name)
                if args_dict is not None:
                    value = args_dict.__getattribute__(name)
                    if isinstance(value, dict):
                        out[name] = module(**value)
                    elif isinstance(value, (list, tuple)):
                        out[name] = module(*value)
                    else:
                        out[name] = module(value)
                else:
                    out[name] = module(*args, **kwargs)

            return out

        module = getattr(self.modules, name)
        res = module(*args, **kwargs)
        return res


class Tmp:
    """Class used as a hack for dot notation access into submodules."""

    def __init__(self, params, apply_fn: Callable):
        self._params = params
        self._apply_fn = apply_fn

    def __call__(self, *args, **kwargs):
        logger.debug("__call__")
        return self._apply_fn(self._params, *args, **kwargs)

    def __getattr__(self, name: str):
        logger.debug("__getattr__")
        return ft.partial(self.__call__, name=name)


class OTrainState(Generic[_N], struct.PyTreeNode):
    """Custom version of flax.training.TrainState but with better type information."""

    step: int
    apply_fn: _OApplyFn = struct.field(pytree_node=False)
    params: _Params
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState | optax.InjectHyperparamsState

    def vars_dict(self, params: _Params | None = None):
        params = get_or(params, self.params)
        return {"params": params}

    def apply_mut(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.apply(*args, **kwargs)

    def apply(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.apply_fn(self.vars_dict(), *args, **kwargs)

    def apply_with(self, *args: _P.args, params: _Params, **kwargs: _P.kwargs) -> _R:
        return self.apply_fn(self.vars_dict(params), *args, **kwargs)

    def apply_gradients(self, grads: _Params, **kwargs) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def set_batch_stats(self, batch_stats: dict) -> "TrainState":
        return self.replace(batch_stats=batch_stats)

    def set_lr(self, lr: FloatScalar):
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                hyperparams[key] = lr
                return
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @property
    def lr(self) -> FloatScalar:
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                return hyperparams[key]
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @property
    def total_notfinite(self) -> int:
        inner_state = self.opt_state.inner_state
        assert isinstance(inner_state, ApplyIfFiniteState)
        return inner_state.total_notfinite

    # @classmethod
    # def create(
    #     cls,
    #     apply_fn: _ApplyFn,
    #     params: _Params,
    #     tx: optax.GradientTransformation,
    #     **kwargs,
    # ) -> "OTrainState":
    #     """Creates a new instance with `step=0` and initialized `opt_state`."""
    #     opt_state = None
    #     if tx is not None:
    #         opt_state = tx.init(params)
    #     return cls(
    #         step=0,
    #         apply_fn=apply_fn,
    #         params=params,
    #         tx=tx,
    #         opt_state=opt_state,
    #         **kwargs,
    #     )

    @classmethod
    def create_from_def(
        cls, key: PRNGKey, net_def: Modules[_N], init_args: dict[str, Any], tx: optax.GradientTransformation, **kwargs
    ) -> "OTrainState":
        logger.debug("init: {}", inspect.signature(net_def.init))
        logger.debug("apply: {}", inspect.signature(net_def.apply))

        variables: dict = net_def.init(key, args_dict=init_args)
        params = variables["params"]
        # return cls.create(net_def.apply, params, tx, **kwargs)

        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)

        return cls(
            step=0,
            apply_fn=net_def.apply,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    @property
    def S(self) -> _N:
        """Helper to access submodules."""
        return Tmp(self.vars_dict(), self.apply_fn)

    def strip(self) -> "TrainState":
        """Remove tx and opt_state."""
        return self.replace(tx=None, opt_state=None)
