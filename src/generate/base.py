import random
from abc import ABC, abstractmethod
from ..data import Note


class SongGenerator(ABC):
    def __init__(self, seed: int | None = None):
        self.randomizer = random.Random(seed)

    @abstractmethod
    def generate(self) -> list[Note]:
        pass
