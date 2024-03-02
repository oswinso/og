import ipdb
import numpy as np

from og.rl.replay_buffer_np import ReplayBufferNp


def test_push_batch():
    item = np.array([1.0])

    rng = np.random.default_rng(seed=12356)
    for ii in range(16):
        # Do the same operations to rb1 and rb2, and make sure that the states are the same.
        # On rb1, use push_batch. On rb2, use push_batch_slow.
        rb1 = ReplayBufferNp.create(item, 5)
        rb2 = ReplayBufferNp.create(item, 5)

        assert np.all(rb1.data == rb2.data)
        assert np.all(rb1.head == rb2.head)
        assert np.all(rb1.size == rb2.size)
        assert np.all(rb1.is_full == rb2.is_full)

        for jj in range(8):
            batch_size = rng.integers(1, 4)
            b_item = rng.uniform(size=(batch_size, 1))

            rb1 = rb1.push_batch(b_item, batch_size)
            rb2 = rb2.push_batch_slow(b_item, batch_size)

            assert np.all(rb1.data == rb2.data)
            assert np.all(rb1.head == rb2.head)
            assert np.all(rb1.size == rb2.size)
            assert np.all(rb1.is_full == rb2.is_full)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test_push_batch()
