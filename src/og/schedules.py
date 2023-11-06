import math

import jax.numpy as jnp
import numpy as np
import optax
from attrs import asdict, define

from og.jax_types import FloatScalar, IntScalar


@define
class Schedule:
    def as_dict(self):
        return {"type": type(self).__name__, **asdict(self)}

    @property
    def total_steps(self) -> int:
        return 0

    def make(self) -> optax.Schedule:
        ...


def as_schedule(val: Schedule | float | int) -> Schedule:
    if isinstance(val, Schedule):
        return val

    return Constant(val)


@define
class Constant(Schedule):
    value: float
    steps: int = 0

    @property
    def total_steps(self):
        return self.steps

    def make(self) -> optax.Schedule:
        return optax.constant_schedule(self.value)
