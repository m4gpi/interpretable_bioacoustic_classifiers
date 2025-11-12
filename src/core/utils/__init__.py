import collections
import torch

from typing import Callable, List

def tree():
    return collections.defaultdict(tree)
