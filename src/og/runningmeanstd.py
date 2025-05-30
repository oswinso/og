import jax.numpy as jnp
from flax import struct

from og.treenode_utils import prettynode


@prettynode
class RunningMeanStd(struct.PyTreeNode):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @staticmethod
    def create(eps: float = 1e-4, mean=None, var=None, count=None) -> "RunningMeanStd":
        mean = mean if mean is not None else 0.0
        var = var if var is not None else 0.0
        count = count if count is not None else eps
        return RunningMeanStd(mean, var, count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> "RunningMeanStd":
        if batch_count == 1:
            batch_var = jnp.zeros_like(batch_mean)

        new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        return self.replace(mean=new_mean, var=new_var, count=new_count)

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    @property
    def std(self):
        return jnp.sqrt(self.var)
