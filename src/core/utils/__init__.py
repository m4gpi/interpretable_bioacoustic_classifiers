import collections
import torch

from typing import Any, Callable, Dict, List

def tree():
    return collections.defaultdict(tree)

def try_or(func: Callable, default: Any) -> Any:
    try:
        return func()
    except Exception as e:
        return default

def to_snake_case(word):
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    return word.lower()

def merge_dicts(d1, d2) -> Dict[Any, Any]:
    return {k: d1.get(k) or d2.get(k) for k in set(list(d1.keys()) + list(d2.keys())) }

def prefix_keys(d: Dict, prefix: str, separator: str = '/') -> Dict[str, Any]:
    return { f"{prefix}{separator}{key}": value for key, value in d.items() }

def detach_values(d: Dict) -> Dict:
    return { k: v.detach() for k, v in d.items() if isinstance(v, torch.Tensor) }
