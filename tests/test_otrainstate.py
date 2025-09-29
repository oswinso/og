from typing import NamedTuple, TypeVar, Generic

import flax.linen as nn
import ipdb
import jax.random as jr
import numpy as np
import optax

from og.train_state import OTrainState, Modules, NT

_M = TypeVar("_M", bound=nn.Module)

# class Wrap(Generic[_M]):
#     def __call__(self, *args, **kwargs):
#         ...


def main():
    key = jr.PRNGKey(0)
    tx = optax.adamw(1e-3)

    hid_size1 = 5
    hid_size2 = 7
    init_args = np.ones(3)

    nn_def1 = nn.Dense(hid_size1)
    nn_def2 = nn.Dense(hid_size2)

    nets_info = {"a": (nn_def1, (init_args,)), "b": (nn_def2, (init_args,))}

    nets = NT(**{k: v[0] for k, v in nets_info.items()})
    args = NT(**{k: v[1] for k, v in nets_info.items()})

    m_def = Modules(nets)
    train_state: OTrainState[NT] = OTrainState.create_from_def(key, m_def, args, tx=tx)

    print(train_state.params)

    out = train_state.S.a(init_args)
    print(out)
    out = train_state.S.b(init_args)
    print(out)
    ipdb.set_trace()

    # train_state = OTrainState.create_from_def(key, nn_def, (init_args,), tx=tx)
    # train_state.S.nn1(args, params=params)
    #
    # out = train_state.apply(init_args)
    # print(out)
    # print(train_state.params)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
