# Using music21, we can make a generator that returns a random Bach chorale from the 371 chorales in the corpus

from music21 import corpus
from .base import SongGenerator
from ..data import _music21_setup, Note
