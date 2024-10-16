import ipdb
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.nnx import bridge

from og.jax_types import FloatScalar
from og.nnx.mlp import LayerNormMLP
from og.nnx.optimizer import Optimizer




def main():
    din = 2
    hid_sizes = [32, 32]
    rngs = nnx.Rngs(12345)
    lr = 1e-3

    nn = LayerNormMLP(
        din=din, hid_sizes=hid_sizes, activation=nnx.relu, rngs=rngs, use_bias=False, use_offset=True, use_scale=True
    )
    tx = optax.adam(lr)
    opt = Optimizer(nn, tx)

    # Compute the initial norms of all the weights.
    initial_norms = [jnp.sqrt(jnp.sum(linear.kernel**2)) for linear in nn.linears]
    projector = Projector(initial_norms)

    # See if the projection preserves the outputs.
    rng = np.random.default_rng(seed=1234567)
    x = rng.uniform(size=(din,))
    y = np.array(nn(x))

    projector(nn)
    y_new = np.array(nn(x))

    np.testing.assert_allclose(y, y_new)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
