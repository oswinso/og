from typing import Generic, TypeVar

import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx.nnx import filterlib, graph
from flax.nnx.nnx.training.optimizer import OptState
from jaxtyping import ArrayLike, Float

_Model = TypeVar("_Model", bound=nnx.Module)


class Optimizer(nnx.Object, Generic[_Model]):
    def __init__(self, model: _Model, tx: optax.GradientTransformation, wrt: filterlib.Filter = nnx.Param):
        self.step = OptState(jnp.array(0, dtype=jnp.uint32))
        self.model = model
        self.tx = tx
        self.opt_state = OptState(tx.init(nnx.state(model, wrt)))
        self.wrt = wrt

    def split(self, *filters: filterlib.Filter):
        return graph.split(self, *filters)

    def update(self, grads):
        state = nnx.state(self.model, self.wrt)

        updates, new_opt_state = self.tx.update(grads, self.opt_state.value, state)
        new_params = optax.apply_updates(state, updates)
        assert isinstance(new_params, nnx.State)

        self.step.value += 1
        nnx.update(self.model, new_params)
        self.opt_state.value = new_opt_state

    @property
    def lr(self) -> Float[ArrayLike, ""]:
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                return hyperparams[key]
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")
