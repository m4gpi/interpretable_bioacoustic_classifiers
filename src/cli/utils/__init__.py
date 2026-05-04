import inspect
import functools
import os
import nemony as nm

@functools.cache
def run_id():
    return os.urandom(6).hex()

def mnemonic(s: str):
    return nm.encode(s, sep='-')

def filter_kwargs_for_callable(callable_obj, kwargs):
    sig = inspect.signature(callable_obj)
    valid_params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in valid_params}
