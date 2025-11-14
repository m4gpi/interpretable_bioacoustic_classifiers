import abc
import attrs
import itertools
import pathlib
import yaml

from typing import Any, Dict, List, Tuple

__all__ = [
    "Hyperparams",
    "SpeciesDetectorHyperparams",
    "TimePoolingHyperparams",
]

class Hyperparams(abc.ABC):
    def combinations(self):
        return []

@attrs.define()
class SpeciesDetectorHyperparams(Hyperparams):
    clf_learning_rate: List[float] = attrs.field(factory=list)
    beta: List[float] = attrs.field(factory=list)
    l1_penalty: List[float] = attrs.field(factory=list)
    penalty_multiplier: List[float] = attrs.field(factory=list)

    def combinations(self) -> List[Dict[str, float]]:
        return [
            dict(zip(["clf_learning_rate", "beta", "l1_penalty", "penalty_multiplier"], combination))
            for combination in itertools.product(self.clf_learning_rate, self.beta, self.l1_penalty, self.penalty_multiplier)
        ]

@attrs.define()
class TimePoolingHyperparams(Hyperparams):
    clf_learning_rate: List[float] = attrs.field(factory=list)
    l1_penalty: List[float] = attrs.field(factory=list)
    pool_method: List[str] = attrs.field(factory=list)

    def combinations(self) -> List[Dict[str, float]]:
        return [
            dict(zip(["clf_learning_rate", "l1_penalty", "pool_method"], combination))
            for combination in itertools.product(self.clf_learning_rate, self.l1_penalty, self.pool_method)
        ]
