"""Timing utilities."""

from typing import Callable

from functools import wraps
from time import time


def timing(f: Callable) -> Callable:
    """Function decorator for timing function calls."""

    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f"func:{f.__name__} args:[{args}, {kwargs}] took: {(te - ts):2.4f} sec")
        return result

    return wrap
