# Using music21, we can make a generator that returns a random Bach chorale from the 371 chorales in the corpus
import os
import music21 as m21
import json
from tqdm import tqdm
from music21 import corpus
from music21.stream.base import Score
from .base import SongGenerator
from ..data import _setup, Note, NotatedTimeNotes, score_to_notes
from functools import lru_cache


@lru_cache(maxsize=1)
def get_chorale_scores(verbose=False) -> list[Score]:
    _setup()
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
        _setup()
        randomizer = self.get_randomizer()
        chorale = randomizer.choice(self._chorales)
        return {
            "soprano": score_to_notes(chorale.parts[0]),
            "alto": score_to_notes(chorale.parts[1]),
            "tenor": score_to_notes(chorale.parts[2]),
            "bass": score_to_notes(chorale.parts[3]),
        }
