# Some basic data structures????

from abc import ABC, abstractmethod
import typing
import numpy as np
import copy

T = typing.TypeVar('T')

class Feature(typing.Generic[T]):
    """Denotes a basic global feature over the whole score with a fixed dimensionality"""
    def __init__(self, feature: list[T]):
        self._feature = copy.deepcopy(feature)
        assert len(self._feature) == self.dims()

    @classmethod
    @abstractmethod
    def dims(cls) -> int:
        """ Returns the dimensionality of the feature."""
        pass

    def __len__(self) -> int:
        return self.dims()

    def __getitem__(self, index: int) -> T:
        return self._feature[index]

class FeatureLocus(typing.Generic[T]):
    def __init__(self, feature: list[list[T]], resolution: float):
        self._feature = copy.deepcopy(feature)
        self._resolution = resolution
        assert len(self._feature) > 0
        assert all(len(f) == len(self._feature[0]) == self.dims() for f in self._feature), "All feature lists must have the same length."

    @classmethod
    @abstractmethod
    def dims(cls) -> int:
        """ Returns the dimensionality of the feature locus."""
        pass

    def __len__(self) -> int:
        return len(self._feature)

    def __getitem__(self, index: int) -> list[T]:
        return self._feature[index]

    @property
    def resolution(self) -> float:
        """Returns the resolution of the feature locus in time."""
        return self._resolution
