from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

import flax.linen as nn
import optax
from flax import struct
from optax._src.wrappers import ApplyIfFiniteState

from og.jax_types import FloatScalar
from og.none import get_or
from og.rng import PRNGKey

_Params = TypeVar("_Params")

_P = ParamSpec("_P")
_R = TypeVar("_R")
_ApplyFn = Callable[Concatenate[_Params, _P], _R]


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
