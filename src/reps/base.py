import os
from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar('T', bound='SymbolicMusic')


class SymbolicMusic(ABC):
    """A base representation class for symbolic music data. All representations must be able to load from xml and save to midi
    as xml keeps basically all information and midi keeps like only the note values"""
    @abstractmethod
    @classmethod
    def load_from_xml(cls: type[T], path: str) -> T:
        """Load a symbolic music representation from an XML file."""
        pass

    @abstractmethod
    def save_to_midi(self, path: str):
        """Save the symbolic music representation to an XML file."""
        pass
