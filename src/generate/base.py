import random
from abc import ABC, abstractmethod
from ..data import NotatedTimeNotes


class SongGenerator(ABC):
    def __init__(self, seed: int | None = None):
        self._seed = seed

    @abstractmethod
    def generate(self) -> NotatedTimeNotes:
        pass

    def get_randomizer(self) -> random.Random:
        return random.Random(self._seed)
