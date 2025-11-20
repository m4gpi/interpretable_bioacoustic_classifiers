import functools
import os

@functools.cache
def run_id():
    return os.urandom(6).hex()


