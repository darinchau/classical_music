# Using music21, we can make a generator that returns a random Bach chorale from the 371 chorales in the corpus
import os
import music21 as m21
import json
from tqdm import tqdm
from music21 import corpus
from .base import SongGenerator
from ..data import _setup, Note, NotatedTimeNotes, score_to_notes
from functools import lru_cache


@lru_cache(maxsize=1)
def get_chorale_scores() -> list[m21.stream.Score]:
    _setup()
    idxs_path = "./resources/valid_chorale_indices.json"
    candidate_chorales = corpus.chorales.ChoraleList().byBudapest

    # if os.path.exists(idxs_path):
    #     with open(idxs_path, "r") as f:
    #         valid_idxs = json.load(f)
    #     scores: list[m21.stream.Score] = []
    #     for i in tqdm(valid_idxs):
    #         score = corpus.parse(f"bwv{candidate_chorales[i]['bwv']}")
    #         assert isinstance(score, m21.stream.Score)
    #         scores.append(score)
    #     print(f"Loaded {len(scores)} chorales")
    #     return scores

    scores: list[m21.stream.Score] = []
    valid_idxs = []
    for i, b in tqdm(candidate_chorales.items(), total=len(candidate_chorales), desc="Loading chorales..."):
        score = corpus.parse(f"bwv{b['bwv']}")
        if not score:
            continue
        if not isinstance(score, m21.stream.Score):
            continue
        if not len(score.parts) == 4:
            continue
        scores.append(score)
        valid_idxs.append(i)

    # with open(idxs_path, "w") as f:
    #     json.dump(sorted(valid_idxs), f)

    print(f"Loaded {len(scores)} chorales.")
    return scores


class ChoraleGenerator(SongGenerator):
    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self._chorales = get_chorale_scores()

    def generate(self) -> NotatedTimeNotes:
        _setup()
        randomizer = self.get_randomizer()
        chorale = randomizer.choice(self._chorales)
        return score_to_notes(chorale)
