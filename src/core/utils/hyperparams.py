import abc
import attrs
import itertools
import pathlib
import yaml

from typing import Any, Dict, List, Tuple

__all__ = [
    "Hyperparams",
    "SpeciesDetectorHyperparams",
]

class Hyperparams(abc.ABC):
    @abc.abstractproperty
    def combinations(self):
        pass

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
