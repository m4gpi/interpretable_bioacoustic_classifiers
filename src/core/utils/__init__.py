import collections
import torch
import numpy as np
import random
import re
import wandb

from torch.functional import F
from typing import Any, Callable, Dict, List, Tuple, NamedTuple

class Batch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor | None = None
    s: torch.Tensor | None = None
    metadata: List[Any] | None = None

    def keys(self):
        return ["x", "y", "s", "metadata"]

    def __getitem__(self, key):
        return {"x": self.x, "y": self.y, "s": self.s, "metadata": self.metadata}[key]

from collections.abc import Mapping
import torch


class TensorDict(dict):
    @staticmethod
    def _zero_like_pair(a, b):
        if a is not None:
            return torch.zeros_like(a)
        if b is not None:
            return torch.zeros_like(b)
        raise ValueError("Cannot infer tensor shape")

    def _binary_op(self, other, op):
        if not isinstance(other, Mapping):
            return NotImplemented
        result = TensorDict()
        all_keys = set(self) | set(other)
        for k in all_keys:
            a = self.get(k)
            b = other.get(k)
            if a is None:
                a = self._zero_like_pair(None, b)
            if b is None:
                b = self._zero_like_pair(a, None)
            result[k] = op(a, b)
        return result

    def __add__(self, other):
        return self._binary_op(other, torch.add)

    def __sub__(self, other):
        return self._binary_op(other, torch.sub)

    def __mul__(self, other):
        # scalar multiply
        if isinstance(other, (int, float)):
            return TensorDict({
                k: v * other
                for k, v in self.items()
            })
        # elementwise dict multiply
        if isinstance(other, Mapping):
            return self._binary_op(other, torch.mul)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return TensorDict({
                k: v / other
                for k, v in self.items()
            })
        if isinstance(other, Mapping):
            return self._binary_op(other, torch.div)
        return NotImplemented

    def clone(self):
        return TensorDict({
            k: v.clone()
            for k, v in self.items()
        })

    def to(self, *args, **kwargs):
        return TensorDict({
            k: v.to(*args, **kwargs)
            for k, v in self.items()
        })

    def detach(self):
        return TensorDict({
            k: v.detach()
            for k, v in self.items()
        })

    def __repr__(self):
        return f"TensorDict({dict.__repr__(self)})"

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        out = TensorDict(self)
        out.update(other)
        return out

    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        out = TensorDict(other)
        out.update(self)
        return out

    def __ior__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        self.update(other)
        return self

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
    return {
        k: (v.detach() if isinstance(v, torch.Tensor) else v)
        for k, v in d.items()
    }

def linear_schedule(
    step: int,
    x_min: float,
    x_max: float,
    warmup_steps: int = 10000,
    hold_steps: int = 10000,
) -> float:
    if step < hold_steps:
        return x_min
    if step >= hold_steps + warmup_steps:
        return x_max
    t = (step - hold_steps) / warmup_steps
    return x_min + (x_max - x_min) * t

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

def bounded_sigmoid(x: float, x_min: float, x_max: float, y_min: float, y_max: float, k: float):
    s = np.floor(np.log10(np.abs(x_max)))
    z = k / 10**(s - 1)
    return y_min + (y_max - y_min) / (1 + np.exp(-z * (x - ((x_min + x_max) / 2))))

def soft_clip(x: torch.Tensor, minimum: int = -6.0) -> torch.Tensor:
    return minimum + F.softplus(x - minimum)

def random_derange(n):
    arr = list(range(n))
    while True:
        random.shuffle(arr)
        if all(arr[i] != i for i in range(n)):
            return arr

def histogram_to_wandb(metrics: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    for k, v in metrics.items():
        if k.endswith("hist"):
            v = wandb.Histogram(np_histogram=v)
        results[k] = v
    return results

def gaussian_kernel(sigmas: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
    sigmas = sigmas.float()
    s_max = sigmas.max()
    c = int(s_max / 0.3 + 1)
    k_size = 2 * c + 1
    x = torch.arange(k_size, device=sigmas.device) - c
    x = x.unsqueeze(0)
    sigmas = sigmas.unsqueeze(1)
    filt = torch.exp(-(x ** 2) / (2 * sigmas ** 2))
    if mask_center:
        filt[:, c] = 0.0
    filt = filt / filt.sum(dim=1, keepdim=True)
    return filt.unsqueeze(1)

def laplace_kernel(sigmas: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
    sigmas = sigmas.float()
    s_max = sigmas.max()
    c = int(s_max / 0.3 + 1)
    k_size = 2 * c + 1
    x = torch.arange(k_size, device=sigmas.device) - c
    x = x.unsqueeze(0)
    sigmas_exp = sigmas.unsqueeze(1)
    filt = torch.exp(-torch.abs(x) / sigmas_exp)
    if mask_center:
        filt[:, c] = 0.0
    filt = filt / filt.sum(dim=1, keepdim=True)
    return filt.unsqueeze(1)
