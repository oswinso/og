from typing import Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, Shaped, PRNGKeyArray

Arr = Union[np.ndarray, Array]

AnyShaped = Shaped[Arr, "*"]
AnyFloat = Float[Arr, "*"]
Shape = Union[tuple[int, ...], list[int]]

FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]

BFloat = Float[Arr, "b"]
BInt = Int[Arr, "b"]
BBool = Bool[Arr, "b"]

RotMat2D = Float[Arr, "2 2"]
RotMat3D = Float[Arr, "3 3"]

FloatDict = dict[str, Union[FloatScalar, "FloatDict"]]
