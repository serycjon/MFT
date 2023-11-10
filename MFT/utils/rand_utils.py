import contextlib
import numpy as np


@contextlib.contextmanager
def tmp_np_seed(seed):
    if seed is None:
        yield  # do nothing
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)
