from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

import flax.linen as nn
from flax import struct

from og.none import get_or
from og.rng import PRNGKey

_Params = TypeVar("_Params")

_P = ParamSpec("_P")
_R = TypeVar("_R")
_ApplyFn = Callable[Concatenate[_Params, _P], _R]


class FixedState(Generic[_R], struct.PyTreeNode):
    apply_fn: _ApplyFn = struct.field(pytree_node=False)
    params: _Params

    def vars_dict(self, params: _Params | None = None):
        params = get_or(params, self.params)
        return {"params": params}

    def apply(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.apply_fn(self.vars_dict(), *args, **kwargs)

    @classmethod
    def create(
        cls,
        apply_fn: _ApplyFn,
        params: _Params,
        **kwargs,
    ) -> "FixedState":
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        return cls(
            apply_fn=apply_fn,
            params=params,
            **kwargs,
        )

    @classmethod
    def create_from_def(cls, key: PRNGKey, net_def: nn.Module, init_args: tuple, **kwargs) -> "FixedState":
        variables: dict = net_def.init(key, *init_args)
        params = variables["params"]
        return cls.create(net_def.apply, params, **kwargs)
