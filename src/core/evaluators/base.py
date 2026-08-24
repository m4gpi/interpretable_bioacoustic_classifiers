import abc
import typing

__all__ = ["Evaluator"]

class Evaluator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args: typing.Any, **kwargs: typing.Any):
        ...
