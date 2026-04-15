import inspect

def filter_kwargs_for_callable(callable_obj, kwargs):
    sig = inspect.signature(callable_obj)
    valid_params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in valid_params}

