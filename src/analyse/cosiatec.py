from dataclasses import dataclass
from collections import defaultdict
import itertools
from ..reps import Note, NotatedTimeNotes

_NoteCoords = tuple[float, int]
TEC = tuple[frozenset[_NoteCoords], set[_NoteCoords]]


def cosiatec(notes: NotatedTimeNotes):
    # Convert notes to points (offset, MIDI_pitch)
    remaining: set[_NoteCoords] = {
        (note.offset, note.midi_number)
        for note in notes
    }

    tecs: list[TEC] = []

    while remaining:
        best_tec = get_best_tec(remaining)
        if not best_tec:
            break

        covered = covered_set(best_tec)
        remaining -= covered
        tecs.append(best_tec)

    # Convert TECs back to note groups
    compressed: list[tuple[list[Note], set[_NoteCoords]]] = []
    for pattern, vectors in tecs:
        pattern_notes = [note for note in notes
                         if (note.offset, note.midi_number) in pattern]
        compressed.append((pattern_notes, vectors))

    return compressed

# Helper functions for COSIATEC implementation


def get_best_tec(points_set: set[_NoteCoords]):
    points = sorted(points_set)
    best = None

    for mtp in find_mtps(points, points_set):
        tec = compute_tec(mtp, points_set)
        conj_tec = compute_conjugate(tec)

        tec = remove_redundant(tec)
        conj_tec = remove_redundant(conj_tec)

        candidates = [tec, conj_tec]
        for candidate in candidates:
            if is_better(candidate, best):
                best = candidate

    return best


def find_mtps(points: list[_NoteCoords], point_set: set[_NoteCoords]):
    vectors: dict[_NoteCoords, list[_NoteCoords]] = defaultdict(list)
    for i, p in enumerate(points):
        for q in points[i+1:]:
            dx = q[0] - p[0]
            dy = q[1] - p[1]
            vectors[(dx, dy)].append(p)

    mtps: list[frozenset[_NoteCoords]] = []
    for vec, starts in vectors.items():
        mtp = [p for p in starts if (p[0]+vec[0], p[1]+vec[1]) in point_set]
        if mtp:
            mtps.append(frozenset(mtp))

    return list({mtp for mtp in mtps if len(mtp) > 1})


def compute_tec(mtp: frozenset[_NoteCoords], points: set[_NoteCoords]) -> TEC:
    translators: set[_NoteCoords] = set()
    p0 = min(mtp)

    for q in points:
        dx = q[0] - p0[0]
        dy = q[1] - p0[1]
        if all((p[0]+dx, p[1]+dy) in points for p in mtp):
            translators.add((dx, dy))

    translators.discard((0, 0))
    return (frozenset(mtp), translators)


def compute_conjugate(tec: TEC) -> TEC:
    pattern, vectors = tec
    if not pattern:
        return (frozenset(), set())

    p0 = min(pattern)
    new_pattern = {p0}
    new_translators: set[_NoteCoords] = set()

    for v in vectors:
        new_pattern.add((p0[0]+v[0], p0[1]+v[1]))

    for p in pattern:
        if p != p0:
            new_translators.add((p[0]-p0[0], p[1]-p0[1]))

    return (frozenset(new_pattern), new_translators)


def remove_redundant(tec: TEC) -> TEC:
    pattern, vectors = tec
    if not vectors:
        return tec

    essential: list[_NoteCoords] = []
    covered = set(pattern)

    for v in sorted(vectors, key=lambda v: (-abs(v[0]), -abs(v[1]))):
        new_points = {(p[0]+v[0], p[1]+v[1]) for p in pattern}
        if not new_points.issubset(covered):
            essential.append(v)
            covered.update(new_points)

    return (pattern, set(essential))


def is_better(a: TEC, b: TEC | None):
    if not b:
        return True
    cr_a = len(covered_set(a)) / (len(a[0]) + len(a[1]))
    cr_b = len(covered_set(b)) / (len(b[0]) + len(b[1]))

    if cr_a != cr_b:
        return cr_a > cr_b
    return len(a[0]) > len(b[0])


def covered_set(tec: TEC):
    if not tec:
        return set()
    pattern, vectors = tec
    covered = set(pattern)
    for v in vectors:
        covered.update((p[0]+v[0], p[1]+v[1]) for p in pattern)
    return covered
