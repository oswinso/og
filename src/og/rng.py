import jax.random as jr
from jaxtyping import PRNGKeyArray

PRNGKey = PRNGKeyArray


class KeyGen:
    key: PRNGKey
    calls: int

    def __init__(self, key: PRNGKey | int):
        if isinstance(key, int):
            key = jr.PRNGKey(key)
        self.key = key
        self.calls = 0

    def __call__(self) -> PRNGKey:
        key = jr.fold_in(self.key, self.calls)
        self.calls += 1
        return key

    def __invert__(self) -> PRNGKey:
        return self.__call__()
