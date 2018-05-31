import contextlib
import numpy as np


@contextlib.contextmanager
def temporary_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def np_contains(l: list, a: np.array) -> bool:
    return any(np.array_equal(a, s) for s in l)
