# Using music21, we can make a generator that returns a random Bach chorale from the 371 chorales in the corpus
import os
import music21 as m21
import json
from tqdm import tqdm
from music21 import corpus
from music21.stream.base import Score, Part, Stream
from .base import SongGenerator
from ..reps import Note, NotatedTimeNotes, Music21Stream
from ..util import _require_music21
from functools import lru_cache


@lru_cache(maxsize=1)
def get_chorale_scores(verbose=False) -> list[Score]:
    _require_music21()
    idxs_path = "./resources/valid_chorale_indices.json"
    candidate_chorales = corpus.chorales.ChoraleList().byBudapest

    scores: list[Score] = []
    valid_idxs = []
    for i, b in tqdm(candidate_chorales.items(), total=len(candidate_chorales), desc="Loading chorales...", disable=not verbose):
        score = corpus.parse(f"bwv{b['bwv']}")
        if not score:
            continue
        if not isinstance(score, Score):
            continue
        if not len(score.parts) == 4:
            continue
        scores.append(score)
        valid_idxs.append(i)
    if verbose:
        print(f"Loaded {len(scores)} chorales.")
    return scores


class ChoraleGenerator(SongGenerator):
    def __init__(self, seed: int | None = None, verbose: bool = False):
        super().__init__(seed)
        self._chorales = get_chorale_scores(verbose=verbose)

    def generate_parts(self) -> dict[str, NotatedTimeNotes]:
        chorale = self.get_chorale()
        return {
            "soprano": score_to_notes(chorale.parts[0]),
            "alto": score_to_notes(chorale.parts[1]),
            "tenor": score_to_notes(chorale.parts[2]),
            "bass": score_to_notes(chorale.parts[3]),
        }

    def get_chorale(self) -> Score:
        """Returns a chorale from the corpus by index"""
        _require_music21()
        randomizer = self.get_randomizer()
        chorale = randomizer.choice(self._chorales)
        return chorale


def score_to_notes(score: Stream) -> NotatedTimeNotes:
    """Converts a music21 score to a NotatedTimeNotes object"""
    return Music21Stream(score).to_notated_time_notes()
