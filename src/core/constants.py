import enum
from typing import Any, Dict

__all__ = [
    "LOG_MEL_SPECTROGRAM_PARAMS",
]

class Stage(enum.Enum):
    TRAIN: int = enum.auto()
    VAL: int = enum.auto()
    TEST: int = enum.auto()
