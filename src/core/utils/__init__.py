import collections
import torch
import numpy as np
import re

from torch.functional import F
from typing import Any, Callable, Dict, List, Tuple

def tree():
    return collections.defaultdict(tree)

def try_or(func: Callable, default: Any) -> Any:
    try:
        return func()
    except Exception as e:
        return default

def to_snake_case(s: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', s)
    s = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', s)
    s = s.replace("-", "_")
    return s.lower()

def to_camel_case(s: str) -> str:
    parts = re.split(r'[^a-zA-Z0-9]+', s.strip().lower())
    return parts[0] + ''.join(word.capitalize() for word in parts[1:] if word)

def to_pascal_case(s: str) -> str:
    parts = re.split(r'[^a-zA-Z0-9]+', s.strip())
    return ''.join(word.capitalize() for word in parts if word)

def merge_dicts(d1, d2) -> Dict[Any, Any]:
    return {k: d1.get(k) or d2.get(k) for k in set(list(d1.keys()) + list(d2.keys())) }

def prefix_keys(d: Dict, prefix: str, separator: str = '/') -> Dict[str, Any]:
    return { f"{prefix}{separator}{key}": value for key, value in d.items() }

def detach_values(d: Dict) -> Dict:
    return { k: v.detach() for k, v in d.items() if isinstance(v, torch.Tensor) }

def nth_percentile(x: torch.Tensor, z_score: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.mean() - z_score * x.std(), x.mean() + z_score * x.std()

def linear_decay(t_current: int, t_start: int, t_end: int, maximum: int, minimum: int):
    beta = max(0.0, 1 - (max(t_start, t_current) - t_start) / (t_end - t_start))
    return minimum + ((maximum - minimum) * beta)

def linear_growth(t_current: int, t_start: int, t_end: int, maximum: int, minimum: int):
    beta = min(1.0, (max(t_start, t_current) - t_start) / (t_end - t_start))
    return minimum + ((maximum - minimum) * beta)

def exponential_growth(t_current: int, t_start: int, t_end: int, maximum: int, minimum: int):
    return 1.0 - np.exp(-((max(t_start, t_current) - t_start) + 1) / (t_end - t_start))

def exponential_decay(t_current: int, t_start: int, t_end: int, maximum: int, minimum: int, decay_rate: float | None = None):
    t_clamped = max(t_start, min(t, t_end))
    decay_rate = decay_rate or -np.log((minimum - maximum) / (maximum - minimum)) / (t_end - t_start)
    return minimum + (maximum - minimum) * np.exp(decay_rate * (t_start - t_clamped))

def bounded_sigmoid(x, x_min, x_max, y_min, y_max, k=1):
    x_rescaled = 12 * (x - x_min) / (x_max - x_min) - 6
    sigmoid = 1 / (1 + np.exp(-k * x_rescaled))
    return y_min + (y_max - y_min) * sigmoid

def soft_clip(x: torch.Tensor, minimum: int = -6.0) -> torch.Tensor:
    return minimum + F.softplus(x - minimum)
