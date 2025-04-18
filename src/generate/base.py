import random
from abc import ABC, abstractmethod
from ..reps import NotatedTimeNotes, Note


class SongGenerator(ABC):
    def __init__(self, seed: int | None = None):
        self._seed = seed

    def generate(self) -> NotatedTimeNotes:
        """Generates a song, returning a NotatedTimeNotes object."""
        song = self.generate_parts()
        notes: list[Note] = []
        for part in song.values():
            notes.extend(part._notes)
        return NotatedTimeNotes(notes=notes)

    @abstractmethod
    def generate_parts(self) -> dict[str, NotatedTimeNotes]:
        """Generates multiple parts of the song, returning a dict of NotatedTimeNotes objects."""
        raise NotImplementedError

    def get_randomizer(self) -> random.Random:
        return random.Random(self._seed)
