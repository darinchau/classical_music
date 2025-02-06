# This module generates synthetic songs for the project
# Code is referenced and modified from this amazing project: https://github.com/JohanGHole/AutomaticCounterpoint

from __future__ import annotations
import copy
import itertools
import json
import math
import matplotlib.pyplot as plt
import random
import re
import time
import typing
from abc import ABC, abstractmethod
from itertools import chain
from ..data import PIANO_A0, PIANO_C8, Note
from .base import SongGenerator

INFINITY = 1 << 64 - 1  # A big number for the search algorithm

ScaleName = typing.Literal["major", "minor"]
SpeciesName = typing.Literal["first", "second", "third", "fourth", "fifth"]
KeyName = typing.Literal[
    'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
    'C#', 'D#', 'F#', 'G#', 'A#',
]
CtpPositionName = typing.Literal["above", "below"]
PartName = typing.Literal["soprano", "alto", "tenor", "bass"]

# Names and ranges
KEY_NAMES: list[KeyName] = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
KEY_NAMES_SHARP: list[KeyName] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

RANGES = {
    "bass": range(40, 65),
    "tenor": range(48, 73),
    "alto": range(53, 78),
    "soprano": range(60, 85)
}

# intervals
P1 = C = Tonic = Unison = 0
m2 = Db = 1
M2 = D = 2
m3 = Eb = 3
M3 = E = 4
P4 = F = 5
d5 = Gb = Tritone = 6
P5 = G = 7
m6 = Ab = 8
M6 = A = 9
m7 = Bb = 10
M7 = B = 11
P8 = Octave = 12

MELODIC_CONSONANT_INTERVALS = [m2, M2, m3, M3, P4, P5, m6, Octave]
MELODIC_INTERVALS = [Unison, m2, M2, m3, M3, P4, P5, m6, P8, -m2, -M2, -m3, -M3, -P4, -P5, -P8]
HARMONIC_DISSONANT_INTERVALS = [m2, M2, P4, M7, m7, P8 + m2, P8 + M2]
HARMONIC_CONSONANCES = [m3, M3, P5, m6, M6, P8, P8 + m3, P8 + M3]
PERFECT_INTERVALS = [P5, P8]


def sign(x):
    return math.copysign(1, x)


def _get_midi_number(pitch: int | str) -> int:
    if isinstance(pitch, int):
        return pitch
    return Note.from_str(pitch).midi_number


def _shift_part(part: PartName, shift: CtpPositionName) -> PartName:
    match (part, shift):
        case ("soprano", "below"):
            return "alto"
        case ("alto", "below"):
            return "tenor"
        case ("tenor", "below"):
            return "bass"
        case ("bass", "above"):
            return "tenor"
        case ("tenor", "above"):
            return "alto"
        case ("alto", "above"):
            return "soprano"
        case (_, _):
            raise ValueError(f"Invalid shift: {part} {shift}")


class Constraints:
    def __init__(self, ctp: Counterpoint):
        self.species = ctp.species
        self.ctp = ctp.melody.pitches
        self.short_cf = ctp.cf.pitches
        self.ctp_rhythm = ctp.melody.rhythm
        self.ctp_flat_rhythm = list(chain.from_iterable(self.ctp_rhythm))
        self.ctp_in_measure = self.convert_pitch_sequence_to_measures()
        self.measure_idx: list[int] = []
        j = 0
        for i in range(len(self.ctp_rhythm)):
            self.measure_idx.append(j)
            j += len(self.ctp_rhythm[i])
        self.scale_pitches = ctp.scale_pitches
        self.cf_notes = self.extend_cf(ctp.cf.pitches.copy())
        self.cf_tonic = self.cf_notes[0]
        self.ties = ctp.melody.ties
        self.ctp_position = ctp.ctp_position
        self.ctp_errors: list[str] = []
        self.start_idx = self._get_numb_start_idx()
        self.end_idx = [len(self.ctp)-1]
        self.weighted_indices = {i: 0 for i in range(len(self.ctp))}

    def extend_cf(self, cf: list[int]):
        cf_extended: list[int] = []
        for m in range(len(self.ctp_rhythm)):
            for n in range(len(self.ctp_rhythm[m])):
                cf_extended.append(cf[m])
        return cf_extended

    def convert_pitch_sequence_to_measures(self):
        pitch_measures: list[list[int]] = []
        i = 0
        for j in range(len(self.ctp_rhythm)):
            measure: list[int] = []
            for note_dur in range(len(self.ctp_rhythm[j])):
                measure.append(self.ctp[i])
                i += 1
            pitch_measures.append(measure)
        return pitch_measures

    def _get_numb_start_idx(self):
        if self.species == 1:
            return [0]
        elif self.species in [2, 3, 4, 5]:
            return [0, 1]
        raise ValueError("Invalid species: {}".format(self.species))

    def motion(self, idx, upper_voice, lower_voice):
        if idx == 0 or upper_voice[0] == -1 or lower_voice[0] == -1:
            return
        cf = upper_voice
        ctp = lower_voice
        cf_dir = cf[idx] - cf[idx - 1]
        ctp_dir = ctp[idx] - ctp[idx - 1]
        if cf_dir == ctp_dir:
            return "parallel"
        elif (cf_dir == 0 and ctp_dir != 0) or (ctp_dir == 0 and cf_dir != 0):
            return "oblique"
        elif sign(cf_dir) == sign(ctp_dir) and cf_dir != ctp_dir:
            return "similar"
        return "contrary"

    def _get_interval_degree(self, interval: int):
        interval_degrees = {
            0: "unison",
            m2: "second", M2: "second",
            m3: "third", M3: "third",
            P4: "fourth",
            P5: "fifth",
            d5: "d5",
            m6: "sixth", M6: "sixth",
            m7: "seventh", M7: "seventh",
            Octave: "octave",
            Octave + m2: "ninth", Octave + M2: "ninth",
            Octave + m3: "tenth", Octave + M3: "tenth"
        }
        if interval in interval_degrees:
            return interval_degrees[interval]
        raise ValueError("Invalid interval: {}".format(interval))

    def _is_large_leap(self, ctp_draft: list[int], idx: int):
        if idx == len(ctp_draft) - 1 or ctp_draft[idx] == -1:
            return False
        if abs(ctp_draft[idx + 1] - ctp_draft[idx]) >= P4:
            return True
        return False

    def _is_small_leap(self, ctp_draft: list[int], idx: int):
        if idx == len(ctp_draft) - 1 or ctp_draft[idx] == -1:
            return False
        if abs(ctp_draft[idx + 1] - ctp_draft[idx]) in [m3, M3]:
            return True
        return False

    def _is_step(self, ctp_draft: list[int], idx: int):
        if idx == len(ctp_draft) - 1 or ctp_draft[idx] == -1:
            return False
        if abs(ctp_draft[idx + 1] - ctp_draft[idx]) in [m2, M2]:
            return True
        return False

    def get_firstbeats(self):
        return self.measure_idx

    def get_downbeats(self):
        indices = list(range(len(self.cf_notes)))
        return indices[::2]

    def get_upbeats(self):
        indices = list(range(len(self.cf_notes)))
        return indices[1::2]

    def _is_melodic_leap_too_large(self, ctp_draft: list[int], idx: int):
        if idx in self.end_idx or ctp_draft[idx] == -1:
            return False
        interval = ctp_draft[idx + 1] - ctp_draft[idx]
        if abs(interval) > P5:
            if self.species == "fifth" and self.ctp_flat_rhythm[idx] < 4 and self.ctp_flat_rhythm[idx+1] < 4:
                return True
            if sign(interval) == 1.0 and interval == m6 and self.species not in ["third"]:
                return False
            if abs(interval) == Octave:
                return False
            return True
        return False

    def _is_melodic_leap_octave(self, ctp_draft: list[int], idx: int):
        if idx in self.end_idx or ctp_draft[idx] == -1:
            return False
        interval = ctp_draft[idx + 1] - ctp_draft[idx]
        return abs(interval) == Octave

    def _is_successive_same_direction_leaps(self, ctp_draft: list[int], idx: int):
        if idx >= self.end_idx[0]-1 or ctp_draft[idx] == -1:
            return False
        interval1 = ctp_draft[idx + 1] - ctp_draft[idx]
        interval2 = ctp_draft[idx + 2] - ctp_draft[idx + 1]
        if abs(interval1) >= m3 and abs(interval2) >= m3:
            if sign(interval1) != sign(interval2):
                return False
            if abs(interval1) + abs(interval2) <= M3 + M3:
                # Outlines a triad, acceptable
                return False
            return True
        return False

    def _is_successive_leaps_valid(self, ctp_draft: list[int], idx: int):
        if idx >= self.end_idx[0]-1 or ctp_draft[idx] == -1:
            return True
        interval1 = ctp_draft[idx + 1] - ctp_draft[idx]
        interval2 = ctp_draft[idx + 2] - ctp_draft[idx + 1]
        if abs(interval1) >= m3 and abs(interval2) >= m3:
            if abs(interval1) + abs(interval2) > Octave:
                return False
            if sign(interval1) == sign(interval2) == 1.0:
                if interval2 > interval1:
                    return False
            if sign(interval1) == sign(interval2) == -1.0:
                if abs(interval1) > abs(interval2):
                    return False
        return True

    def _is_leap_compensated(self, ctp_draft: list[int], idx: int):
        if idx >= self.end_idx[0] - 1 or ctp_draft[idx] == -1:
            return True
        interval1 = ctp_draft[idx + 1] - ctp_draft[idx]
        interval2 = ctp_draft[idx + 2] - ctp_draft[idx + 1]
        if abs(interval1) > P5:
            if sign(interval1) == 1.0 and sign(interval2) == -1.0 and abs(interval2) <= M2:
                return True
            elif sign(interval1) == -1.0 and sign(interval2) == 1.0 and abs(interval2) <= M3:
                return True
            else:
                return False
        return True

    def _is_octave_compensated(self, ctp_draft: list[int], idx: int):
        if idx >= self.end_idx[0] - 2 or ctp_draft[idx] == -1:
            return True
        interval1 = ctp_draft[idx + 1] - ctp_draft[idx]
        interval2 = ctp_draft[idx + 2] - ctp_draft[idx + 1]
        if self._is_melodic_leap_octave(ctp_draft, idx + 1):
            return self._is_leap_compensated(ctp_draft, idx + 1) and sign(interval1) != sign(interval2)
        return True

    def _is_chromatic_step(self, ctp_draft: list[int], idx: int):
        if idx >= self.end_idx[0] or ctp_draft[idx] == -1:
            return False
        if abs(ctp_draft[idx + 1] - ctp_draft[idx]) == 1 and ctp_draft[idx + 1] not in self.scale_pitches:
            return True
        return False

    def _is_repeating_pitches(self, ctp_draft: list[int], idx: int):
        if idx in self.end_idx:
            return False
        if ctp_draft[idx] == ctp_draft[idx + 1]:
            return not self.ties[idx]
        return False

    def _is_within_range_of_a_tenth(self, ctp_draft: list[int]):
        return max(ctp_draft) - min(ctp_draft[1:]) <= Octave + M3

    def _is_unique_climax(self, ctp_draft: list[int]):
        # Unique climax that is different from the cantus firmus or with sufficient spacing
        climax = max(ctp_draft)
        climax_measure_idx = []
        for measure in range(len(self.ctp_in_measure)):
            if climax in self.ctp_in_measure[measure]:
                for i in range(self.ctp_in_measure[measure].count(climax)):
                    climax_measure_idx.append(measure)
        if len(climax_measure_idx) == 1:
            if self.short_cf.index(max(self.short_cf)) == climax_measure_idx[0]:
                return False
            else:
                return True
        for i in range(len(climax_measure_idx)-1):
            if abs(climax_measure_idx[i] - climax_measure_idx[i+1]) < 4:
                return False
            if self.short_cf.index(max(self.short_cf)) == climax_measure_idx[i]:
                return False
        return True

    def _is_leading_tone_properly_resolved(self, ctp_draft: list[int]):
        return abs(ctp_draft[-1] - ctp_draft[-2]) in [m2, M2]

    def _is_motivic_repetitions(self, ctp_draft: list[int], idx: int):
        if idx >= len(ctp_draft)-3:
            return False
        if ctp_draft[idx:idx + 2] == ctp_draft[idx + 2:idx + 4]:
            return True
        return False

    def _melodic_rules(self, ctp_draft: list[int]):
        penalty = 0
        # Index based rules
        if self.species >= 1:  # valid melodic rules for each species
            for i in range(len(ctp_draft)):
                if self._is_melodic_leap_too_large(ctp_draft, i):
                    self.ctp_errors.append("Too large leap!")
                    penalty += 100
                    self.weighted_indices[i] += 4
                if self._is_melodic_leap_octave(ctp_draft, i):
                    self.ctp_errors.append("Octave leap!")
                    penalty += 25
                    self.weighted_indices[i] += 1
                if not self._is_leap_compensated(ctp_draft, i):
                    self.ctp_errors.append("Leap not compensated!")
                    penalty += 50
                    self.weighted_indices[i] += 2
                if not self._is_octave_compensated(ctp_draft, i):
                    self.ctp_errors.append("Octave not compensated!")
                    penalty += 25
                    self.weighted_indices[i] += 1
                if self._is_successive_same_direction_leaps(ctp_draft, i):
                    self.ctp_errors.append(
                        "Successive Leaps in same direction!")
                    penalty += 25
                    self.weighted_indices[i] += 1
                    if not self._is_successive_leaps_valid(ctp_draft, i):
                        self.ctp_errors.append(
                            "Successive leaps strictly not valid!")
                        penalty += 100
                        self.weighted_indices[i] += 4
                if self._is_chromatic_step(ctp_draft, i):
                    self.ctp_errors.append("Chromatic movement!")
                    penalty += 100
                    self.weighted_indices[i] += 4
                if self._is_repeating_pitches(ctp_draft, i):
                    self.ctp_errors.append("Repeats pitches!")
                    penalty += 100
                    self.weighted_indices[i] += 1
            # Global rules
            if not self._is_within_range_of_a_tenth(ctp_draft):
                self.ctp_errors.append("Exceeds the range of a tenth!")
                penalty += 50
            if not self._is_unique_climax(ctp_draft):
                self.ctp_errors.append(
                    "No unique climax or at same position as other voices!")
                penalty += 100
            if not self._is_leading_tone_properly_resolved(ctp_draft):
                self.ctp_errors.append("leading tone not properly resolved!")
                penalty += 100
        if self.species >= 2:
            for i in range(len(ctp_draft)):
                if self._is_motivic_repetitions(ctp_draft, i):
                    self.ctp_errors.append("Motivic repetitions!")
                    penalty += 100
        return penalty

    def _is_perfect_interval_properly_approached(self, upper_voice, lower_voice, idx):
        # the start and end notes are allowed to be perfect
        if idx in self.start_idx or idx in self.end_idx:
            return True
        # always checked between the strongest measure beat
        # if the index is not on a strong beat, it is therefore accepted
        if idx not in self.measure_idx:
            return True
        if upper_voice[idx] - lower_voice[idx] in PERFECT_INTERVALS:
            # The current harmonic interval is perfect
            if self.motion(idx, upper_voice, lower_voice) not in ["oblique", "contrary"]:
                # if the harmonic interval is not approached
                # by oblique or contrary motion, it is not valid
                return False
            if self._is_large_leap(upper_voice, idx - 1) or \
                    self._is_large_leap(lower_voice, idx - 1):
                if upper_voice[idx] - lower_voice[idx] == Octave:
                    # Octave must be approached by oblique motion
                    if self.motion(idx, upper_voice, lower_voice) == "oblique":
                        return True
                else:
                    return False
        return True

    def _is_valid_consecutive_perfect_intervals(self, upper_voice, lower_voice, idx):
        if upper_voice[idx] == -1 or lower_voice[idx] == -1 or idx in self.end_idx or idx not in self.measure_idx:
            return True
        harm_int1 = upper_voice[idx] - lower_voice[idx]
        harm_int2 = upper_voice[idx + 1] - lower_voice[idx + 1]
        if harm_int1 in PERFECT_INTERVALS and harm_int2 in PERFECT_INTERVALS:
            return self._is_step(upper_voice, idx) or self._is_step(lower_voice, idx)
        return True

    def _is_parallel_fourths(self, upper_voice, lower_voice, idx):
        if upper_voice[idx] == -1 or lower_voice[idx] == -1 or idx == self.end_idx[0]:
            return False
        if self.motion(idx, upper_voice, lower_voice) == "parallel" and upper_voice[idx] - lower_voice[idx] == P4:
            return True
        return False

    def _is_voice_overlapping(self, upper_voice, lower_voice, idx):
        if upper_voice[idx] == -1 or lower_voice[idx] == -1:
            return False
        if idx < self.end_idx[0] and lower_voice[idx + 1] >= upper_voice[idx]:
            return True
        return False

    def _is_voice_crossing(self, upper_voice, lower_voice, idx):
        if upper_voice[idx] == -1 or lower_voice[idx] == -1:
            return False
        if upper_voice[idx] - lower_voice[idx] < 0:
            return True
        return False

    def _is_contrary_motion(self, upper_voice, lower_voice, idx):
        return self.motion(idx, upper_voice, lower_voice) == "contrary"

    def _is_valid_number_of_consecutive_intervals(self, upper_voice, lower_voice):
        valid = True
        for i in range(len(lower_voice) - 3):
            i1 = self._get_interval_degree(upper_voice[i] - lower_voice[i])
            i2 = self._get_interval_degree(
                upper_voice[i + 1] - lower_voice[i + 1])
            i3 = self._get_interval_degree(
                upper_voice[i + 2] - lower_voice[i + 2])
            i4 = self._get_interval_degree(
                upper_voice[i + 3] - lower_voice[i + 3])
            if i1 == i2 == i3 == i4:
                valid = False
        return valid

    def _is_unisons_between_terminals(self, ctp):
        num = 0
        for i in range(1, len(ctp)-1):
            if ctp[i] == self.cf_tonic:
                if self.species in [3, 5]:
                    if i in self.measure_idx:
                        num += 1
                    else:
                        num += 0
                else:
                    num += 1
        return num

    def _is_parallel_perfects_on_downbeats(self, ctp_draft, upper_voice, lower_voice):
        if self.species == 2 or self.species == 4:
            db = self.get_downbeats()
        else:
            db = self.get_firstbeats()
        for i in range(len(db) - 1):
            interval1 = upper_voice[db[i]] - lower_voice[db[i]]
            interval2 = upper_voice[db[i + 1]] - lower_voice[db[i + 1]]
            if interval1 == interval2 and interval1 in PERFECT_INTERVALS:
                # consecutive perfects on downbeats
                if upper_voice[db[i + 1]] - upper_voice[db[i]] == lower_voice[db[i + 1]] - lower_voice[db[i]]:
                    # consecutive and parallel
                    return ctp_draft[db[i]] - ctp_draft[db[i] + 1] <= M3
                return False
        return False

    def _voice_independence_rules(self, ctp_draft, cf_notes):
        if self.ctp_position == "above":
            upper_voice = ctp_draft
            lower_voice = cf_notes
        else:
            upper_voice = cf_notes
            lower_voice = ctp_draft
        penalty = 0
        # Index based rules
        if self.species >= 1:  # valid rules for each species
            for i in range(len(ctp_draft)):
                if not self._is_perfect_interval_properly_approached(upper_voice, lower_voice, i):
                    self.ctp_errors.append(
                        "Perfect interval not properly approached!")
                    penalty += 100
                    self.weighted_indices[i] += 4
                if not self._is_valid_consecutive_perfect_intervals(upper_voice, lower_voice, i):
                    self.ctp_errors.append(
                        "Consecutive perfect intervals, but they are not valid!")
                    penalty += 100
                    self.weighted_indices[i] += 4
                if self._is_parallel_fourths(upper_voice, lower_voice, i):
                    self.ctp_errors.append("Parallel fourths!")
                    penalty += 50
                    self.weighted_indices[i] += 2
                if self._is_voice_overlapping(upper_voice, lower_voice, i):
                    self.ctp_errors.append("Voice Overlapping!")
                    penalty += 100
                    self.weighted_indices[i] += 4
                if self._is_voice_crossing(upper_voice, lower_voice, i):
                    self.ctp_errors.append("Voice crossing!")
                    penalty += 50
                    self.weighted_indices[i] += 2
                if self._is_contrary_motion(upper_voice, lower_voice, i):
                    # This not not a severe violation, but more of a preference to avoid similar motion
                    penalty += 5
            # Global rules
            if not self._is_valid_number_of_consecutive_intervals(upper_voice, lower_voice):
                self.ctp_errors.append("Too many consecutive intervals!")
                penalty += 100
            if self._is_unisons_between_terminals(ctp_draft) > 0:
                self.ctp_errors.append("Unison between terminals!")
                penalty += 50 * self._is_unisons_between_terminals(ctp_draft)
        if self.species >= 2:
            if self._is_parallel_perfects_on_downbeats(ctp_draft, upper_voice, lower_voice):
                self.ctp_errors.append(
                    "Parallel perfect intervals on downbeats!")
                penalty += 100
        return penalty

    def _is_dissonant_interval(self, upper_voice, lower_voice, idx):
        return (upper_voice[idx]-lower_voice[idx]) in HARMONIC_DISSONANT_INTERVALS

    def _is_dissonance_properly_left_and_approached(self, idx, ctp_draft):
        current_note = ctp_draft[idx]
        prev_note = ctp_draft[idx-1]
        next_note = ctp_draft[idx+1]
        if abs(next_note-current_note) <= M2 and abs(current_note-prev_note) <= M2:
            if self.species in [3, 5]:
                return True
            return sign(next_note-current_note) == sign(next_note-current_note)
        return False

    def _tied_note_properly_resolved(self, cf_notes, ctp_draft):
        penalty = 0
        if self.ctp_position == "above":
            upper = ctp_draft
            lower = cf_notes
        else:
            upper = cf_notes
            lower = ctp_draft
        for i in range(len(ctp_draft)-1):
            if i in self.start_idx or i in self.end_idx:
                penalty += 0
            elif i in self.get_downbeats():
                if upper[i]-lower[i] in HARMONIC_CONSONANCES:
                    penalty += 0
                else:
                    if (ctp_draft[i+1] - ctp_draft[i]) < 0 and self._is_step(ctp_draft, i):
                        penalty += 0
                    else:
                        self.ctp_errors.append(
                            "Dissonance not properly resolved")
                        penalty += 100
        return penalty

    " Fifth species"

    def _is_eight_note_handled(self, idx, ctp_draft):
        if idx in self.start_idx or idx in self.end_idx:
            return True
        if self.ctp_flat_rhythm[idx] != 1:
            # Not an eight note
            return True
        # The eight note is not approached or left by step
        return self._is_step(ctp_draft, idx) and self._is_step(ctp_draft, idx-1)

    def _dissonance_rules(self, cf_notes, ctp_draft):
        penalty = 0
        if self.species == 1:
            # In first species there is no dissonance,
            # so the allowed harmonic intervals are consonances
            return penalty
        if self.ctp_position == "above":
            upper = ctp_draft
            lower = cf_notes
        else:
            upper = cf_notes
            lower = ctp_draft
        if self.species in [2, 3, 5]:
            for i in range(1, len(ctp_draft)-1):
                if self.species in [3, 5] and self._is_cambiata(i, cf_notes, ctp_draft):
                    # allowed
                    penalty += 0
                elif self._is_dissonant_interval(upper, lower, i):
                    if not self._is_dissonance_properly_left_and_approached(i, ctp_draft):
                        self.ctp_errors.append(
                            "Dissonance not properly left or approached!")
                        penalty += 100
        if self.species == 5:
            for i in range(1, len(ctp_draft)-1):
                if not self._is_eight_note_handled(i, ctp_draft):
                    self.ctp_errors.append("eight notes not properly handled!")
                    penalty += 100
        if self.species in [4, 5]:
            self.ctp_errors.append("tied notes not properly handled!")
            penalty += self._tied_note_properly_resolved(cf_notes, ctp_draft)

        return penalty

    def _is_cambiata(self, idx, cf_notes, ctp_draft):
        if idx >= len(self.ctp_flat_rhythm)-4:
            return False
        if idx in self.measure_idx:
            if self.ctp_rhythm[self.measure_idx.index(idx)] == (2, 2, 2, 2):
                notes = [ctp_draft[idx+i] for i in range(4)]
                cf = cf_notes[idx]
                if self.ctp_position == "above":
                    intervals = [note - cf for note in notes]
                    interval_degree = []
                    for i in intervals:
                        interval_degree.append(self._get_interval_degree(i))
                    if interval_degree == ["octave", "seventh", "fifth", "sixth"]:
                        return True
                    if interval_degree == ["sixth", "d5", "third", "fourth"]:
                        return True
                else:
                    intervals = [cf-note for note in notes]
                    interval_degree = []
                    for i in intervals:
                        interval_degree.append(self._get_interval_degree(i))
                    if interval_degree == ["third", "fourth", "sixth", "fifth"]:
                        return True
        return False

    def _is_valid_terminals(self, ctp_draft, cf_notes):
        # check start and end pitches and see if they are valid
        # must begin and end with perfect consonances (octaves, fifths or unison)
        # Octaves or unisons preferred at the end (i.e. perfect fifth not allowed)
        # if below, the start and end must be the octave the cf
        if ctp_draft[0] == -1:
            idx = 1
        else:
            idx = 0
        if self.ctp_position == "above":
            return not (
                (ctp_draft[idx] - cf_notes[idx] not in [Unison, P5, Octave]) or (
                    ctp_draft[-1] - cf_notes[-1] not in [Unison, Octave])
            )
        return not (
            (cf_notes[idx] - ctp_draft[idx] not in [Unison, Octave]) or (
                cf_notes[-1] - ctp_draft[-1] not in [Unison, Octave])
        )

    def _no_outlined_tritone(self, ctp_draft: list[int]):
        outline_idx = [0]
        outline_intervals = []
        not_allowed_intervals = [Tritone]
        # mellom ytterkant og inn + endring innad
        dir = [sign(ctp_draft[i + 1] - ctp_draft[i])
               for i in range(1, len(ctp_draft) - 1)]
        for i in range(1, len(dir) - 1):
            if dir[i] != dir[i + 1]:
                outline_idx.append(i + 1)
        outline_idx.append(len(ctp_draft) - 1)
        # Iterate over the outline indices and check if a tritone is found
        for i in range(len(outline_idx) - 1):
            outline_intervals.append(
                abs(ctp_draft[outline_idx[i]] - ctp_draft[outline_idx[i + 1]]))

        for interval in not_allowed_intervals:
            if interval in outline_intervals:
                return False

        return True

    def _harmonic_rules(self, ctp_draft, cf_notes):
        penalty = 0
        if self.species >= 1:  # valid harmonic rules for each species
            if not self._is_valid_terminals(ctp_draft, cf_notes):
                self.ctp_errors.append("Terminals not valid!")
                penalty += 100
        if self.species in [3, 5]:
            if not self._no_outlined_tritone(ctp_draft):
                self.ctp_errors.append("Outlined dissonant interval!")
                penalty += 50
        return penalty

    def cost_function(self):
        """Main cost function that calculates the total penalty"""
        penalty = 0
        self.ctp_errors = []
        ctp_draft = self.ctp
        cf_notes = self.cf_notes
        penalty += self._melodic_rules(ctp_draft)
        penalty += self._voice_independence_rules(ctp_draft, cf_notes)
        penalty += self._dissonance_rules(cf_notes, ctp_draft)
        penalty += self._harmonic_rules(ctp_draft, cf_notes)
        return penalty

    def get_errors(self):
        return self.ctp_errors

    def get_weighted_indices(self):
        return dict(sorted(self.weighted_indices.items(), reverse=True, key=lambda item: item[1]))


class Scale:
    """
    The scale class manages scale object. This include the construction of scales included in the NAMED_SCALE dict.
    the scale is a list of all possible notes in the given scale across the entire piano. This means that the root note
    is not necessarily the lowest note.
    """

    def __init__(self, key: KeyName, scale: ScaleName, scale_range: range | None = None):
        # Get the lowest octave
        oct = 0 if key in ["A", "A#", "B", "Bb"] else 1
        self.root: int = _get_midi_number(key + str(oct))  # sets the root of the scale in valid string format
        self.key: KeyName = key
        self.scale_type: ScaleName = scale
        self.intervals = Scale.intervals_from_name(scale)
        self.scale: list[int] = self.build_scale()
        self.scale_pitches: list[int] = self.get_scale_pitches()
        if scale_range is not None:
            self.limit_range(scale_range)

    @staticmethod
    def intervals_from_name(scale_name: ScaleName) -> tuple[int, ...]:
        return tuple({
            "major": (2, 2, 1, 2, 2, 2, 1),
            "minor": (2, 1, 2, 2, 1, 2, 2),
        }[scale_name])

    def build_scale(self) -> list[int]:
        """Builds the scale from the root note"""
        start_pitch = self.root
        scale_len = len(self.intervals)
        highest_possible_pitch = PIANO_C8
        lowest_possible_pitch = PIANO_A0
        j = 0
        scale: list[int] = []
        pitch = start_pitch
        # adds all possible values above the root pitch
        while pitch <= highest_possible_pitch:
            scale.append(_get_midi_number(pitch))
            pitch = scale[j] + self.intervals[j % scale_len]
            j += 1
        # adds all possible values under the root pitch
        j = scale_len - 1
        pitch = start_pitch - self.intervals[j % scale_len]
        while pitch >= lowest_possible_pitch:
            scale.insert(0, _get_midi_number(pitch))
            j -= 1
            pitch = pitch - self.intervals[j % scale_len]
        return scale

    def get_scale_pitches(self) -> list[int]:
        """Get the midi numbers of all notes in the scale"""
        scale_pitches: list[int] = []
        for note in self.scale:
            scale_pitches.append(note)
        return scale_pitches

    def get_scale_range(self, scale_range: list[int]) -> list[int]:
        """Limits the scale into the given scale range (computes the intersection) and returns the midi numbers"""
        scale_pitches: list[int] = []
        for note in scale_range:
            if note in self.scale_pitches:
                scale_pitches.append(note)
        return scale_pitches

    def limit_range(self, scale_range: typing.Sequence[int]) -> None:
        scale: list[int] = []
        for note in scale_range:
            if note in self.scale_pitches:
                scale.append(_get_midi_number(note))
        self.scale = scale

    def set_time(self, duration: float) -> None:
        raise NotImplementedError


class Melody:
    def __init__(
            self,
            key: KeyName,
            scale: ScaleName,
            bar_length: float,
            melody_notes: list[int] | None = None,
            melody_rhythm: list[list[int]] | None = None,
            ties: list[bool] | None = None,
            start: int = 0,
            voice_range: range = RANGES["alto"]
    ):
        self.key: KeyName = key
        self.scale_name: ScaleName = scale
        self.voice_range = voice_range
        self.scale = Scale(key, scale, voice_range)
        self.scale_pitches = self.scale.get_scale_pitches()
        self.note_resolution: int = 8
        self.start = start
        self.bar_length = float(bar_length)

        self._pitches: list[int] | None = melody_notes
        self._rhythm: list[list[int]] | None = melody_rhythm
        self._ties: list[bool] | None = ties
        if self._pitches is not None:
            self.search_domain: list[list[int]] = [self.scale_pitches for _ in self._pitches]
        else:
            self.search_domain: list[list[int]] = [self.scale_pitches]

    def set_ties(self, ties: list[bool]) -> None:
        self._ties = ties.copy()

    def set_rhythm(self, rhythm: list[list[int]]) -> None:
        self._rhythm = copy.deepcopy(rhythm)

    def set_melody(self, melody: list[int]) -> None:
        self._pitches = melody.copy()

    @property
    def pitches(self) -> list[int]:
        return self._pitches.copy() if self._pitches is not None else []

    @property
    def rhythm(self) -> list[list[int]]:
        return copy.deepcopy(self._rhythm) if self._rhythm is not None else []

    @property
    def ties(self) -> list[bool]:
        return self._ties.copy() if self._ties is not None else []

    def get_end_time(self) -> float:
        raise NotImplementedError

    def to_list_notes(self, start: float = 0.) -> list[Note]:
        i = 0
        measure = 0
        t = start
        notes: list[Note] = []
        while measure < len(self.rhythm):
            note_duration = 0
            while note_duration < len(self.rhythm[measure]):
                dur = self.rhythm[measure][note_duration]
                duration = float(dur*self.bar_length / float(self.note_resolution))
                if self.ties[i] == True:
                    measure += 1
                    note_duration = 0
                    dur = self.rhythm[measure][note_duration]
                    duration += float(dur*self.bar_length / float(self.note_resolution))
                    i += 1
                if self.pitches[i] != -1:
                    note = Note.from_midi_number(
                        self.pitches[i],
                        duration=duration,
                        offset=t,
                        real_time=False,
                    )
                    notes.append(note)
                t += duration
                i += 1
                note_duration += 1
            measure += 1
        return notes


class CantusFirmus(Melody):
    PERFECT_INTERVALS: list[int] = [Unison, P5, Octave]
    DISSONANT_INTERVALS: list[int] = [m7, M7, Tritone, -m6, -m7, -M7]
    CONSONANT_MELODIC_INTERVALS: list[int] = [m2, M2, m3, M3, P4, P5, m6, Octave, -m2, -M2, -m3, -M3, -P4, -P5, -Octave]

    def __init__(
            self,
            key: KeyName,
            scale: ScaleName,
            bar_length: float,
            melody_notes: list[int] | None = None,
            melody_rhythm: list[list[int]] | None = None,
            start: int = 0,
            part: PartName = "alto",
            randomizer: random.Random | None = None
    ):
        voice_range = RANGES[part]
        super(CantusFirmus, self).__init__(
            key=key,
            scale=scale,
            bar_length=bar_length,
            melody_notes=melody_notes,
            melody_rhythm=melody_rhythm,
            start=start,
            voice_range=voice_range
        )
        self.part: PartName = part
        self.cf_errors: list[str] = []
        self.rhythm: list[tuple[int]] = self._generate_rhythm()
        self.ties: list[bool] = [False] * len(self.rhythm)
        self.pitches: list[int] = self._generate_cf()
        self.length: int = len(self.rhythm)
        self.randomizer = randomizer if randomizer is not None else random.Random()

    def _start_note(self) -> tuple[list[int], int]:
        if self.key in KEY_NAMES:
            root_idx: int = KEY_NAMES.index(self.key)
        else:
            root_idx: int = KEY_NAMES_SHARP.index(self.key)
        possible_start_notes: list[int] = []
        for pitches in self.voice_range:
            if pitches % Octave == root_idx:
                possible_start_notes.append(pitches)
        tonics: list[int] = possible_start_notes
        return tonics, possible_start_notes[0]

    def _penultimate_note(self) -> int:
        """ The last note can be approached from above or below.
            It is however most common that the last note is approached from above
        """
        leading_tone: int = self._start_note()[1] - 1
        super_tonic: int = self._start_note()[1] + 2
        weights: list[float] = [0.1, 0.9]  # it is more common that the penultimate note is the supertonic than leading tone
        penultimate_note: int = self.randomizer.choices([leading_tone, super_tonic], weights)[0]
        return penultimate_note

    def _get_leading_tones(self) -> int:
        if self.scale_name == "minor":
            leading_tone: int = self._start_note()[1] - 2
        else:
            leading_tone = self._start_note()[1] - 1
        return leading_tone

    def _generate_rhythm(self) -> list[tuple[int]]:
        """
        Generates a random rhythm for the cantus firmus
        Empirically, 12 seems to be the most common, but the rhythm can be any length between 8 and 14
        """
        random_length: int = self.randomizer.choice(list(range(8, 15)) + [12] * 2)
        return [(8,)] * random_length

    def _is_step(self, note: int, prev_note: int) -> bool:
        return abs(prev_note - note) in [m2, M2]

    def _is_small_leap(self, note: int, prev_note: int) -> bool:
        return abs(prev_note - note) in [m3, M3]

    def _is_large_leap(self, note: int, prev_note: int) -> bool:
        return abs(prev_note - note) >= P4

    def _is_climax(self, cf_shell: list[int]) -> bool:
        return cf_shell.count(max(cf_shell)) == 1

    def _is_resolved_leading_tone(self, cf_shell: list[int]) -> bool:
        tonics = self._start_note()[0]
        leading_tone = self._get_leading_tones()
        return not (leading_tone in cf_shell and cf_shell[cf_shell.index(leading_tone)+1] != tonics[0])

    def _is_dissonant_intervals(self, cf_shell: list[int]) -> bool:
        for i in range(len(cf_shell)-1):
            if cf_shell[i+1] - cf_shell[i] in self.DISSONANT_INTERVALS:
                return True
        return False

    def _check_leaps(self, cf_shell: list[int]) -> int:
        penalty: int = 0
        num_large_leaps: int = 0
        for i in range(len(cf_shell)-2):
            if self._is_large_leap(cf_shell[i], cf_shell[i+1]):
                num_large_leaps += 1
                if abs(cf_shell[i]-cf_shell[i+1]) == Octave:
                    # small penalty for octave leap
                    self.cf_errors.append("penalty for octave leap")
                    penalty += 50
                # Check consecutive leaps first
                elif self._is_large_leap(cf_shell[i+1], cf_shell[i+2]):
                    self.cf_errors.append("consecutive leaps")
                    penalty += 25
                elif self._is_large_leap(cf_shell[i+1], cf_shell[i+2]) and sign(cf_shell[i+1]-cf_shell[i]) != sign(cf_shell[i+2]-cf_shell[i+1]):
                    self.cf_errors.append("Large leaps in opposite direction")
                    penalty += 75
                elif self._is_step(cf_shell[i+1], cf_shell[i+2]) and sign(cf_shell[i+1]-cf_shell[i]) == sign(cf_shell[i+2]-cf_shell[i+1]):
                    self.cf_errors.append("A leap is not properly recovered")
                    penalty += 75
        if num_large_leaps >= int(len(self.rhythm) / 2) - 2:
            penalty += 100
        return penalty

    def _is_valid_note_count(self, cf_shell: list[int]) -> bool:
        return not any(cf_shell.count(notes) > 4 for notes in set(cf_shell))

    def _is_valid_range(self, cf_shell: list[int]) -> bool:
        return not (abs(max(cf_shell) - min(cf_shell)) > Octave + M3)

    def _is_repeated_motifs(self, cf_shell: list[int]) -> bool:
        paired_notes: list[list[int]] = []
        for i in range(len(cf_shell)-1):
            if cf_shell[i] == cf_shell[i+1]:
                return True
            if cf_shell[i] == cf_shell[0] and i != 0:
                return True
            paired_notes.append([cf_shell[i], cf_shell[i+1]])
        for pairs in paired_notes:
            if paired_notes.count(pairs) > 1:
                return True
        return False

    def _cost_function(self, cf_shell: list[int]) -> int:
        penalty: int = 0
        penalty = self._check_leaps(cf_shell)
        if not self._is_valid_note_count(cf_shell):
            self.cf_errors.append("note repetition")
            penalty += 100
        if not self._is_climax(cf_shell):
            self.cf_errors.append("no unique cf climax")
            penalty += 100
        if not self._is_valid_range(cf_shell):
            self.cf_errors.append("exceeds the range of a tenth")
            penalty += 100
        if self._is_repeated_motifs(cf_shell):
            self.cf_errors.append("motivic repetitions")
            penalty += 100
        if not self._is_resolved_leading_tone(cf_shell):
            self.cf_errors.append("leading tone not resolved")
            penalty += 100
        if self._is_dissonant_intervals(cf_shell):
            self.cf_errors.append("dissonant interval")
            penalty += 100
        return penalty

    def _initialize_cf(self) -> list[int]:
        # Generates a random cantus firmus shell
        start_note: int = self._start_note()[1]
        end_note: int = start_note
        penultimate_note: int = self._penultimate_note()
        length: int = len(self.rhythm)
        cf_shell: list[int] = [self.randomizer.choice(self.scale_pitches) for _ in range(length)]
        cf_shell[0] = start_note
        cf_shell[-1] = end_note
        cf_shell[-2] = penultimate_note
        return cf_shell

    def _get_melodic_consonances(self, prev_note: int) -> list[int]:
        mel_cons: list[int] = []
        for intervals in self.CONSONANT_MELODIC_INTERVALS:
            if prev_note+intervals in self.scale_pitches:
                mel_cons.append(prev_note+intervals)
        # To further randomize the generated results, the melodic consonances are shuffled
        self.randomizer.shuffle(mel_cons)
        return mel_cons

    def _generate_cf(self) -> list[int]:
        total_penalty: int = INFINITY
        iteration: int = 0
        cf_shell: list[int] = self._initialize_cf()
        while total_penalty > 0:
            cf_shell: list[int] = self._initialize_cf()
            for i in range(1, len(cf_shell)-2):
                self.cf_errors = []
                local_max: int = INFINITY
                cf_draft: list[int] = cf_shell.copy()
                mel_cons: list[int] = self._get_melodic_consonances(cf_shell[i-1])
                best_choice: int = -1
                for notes in mel_cons:
                    cf_draft[i] = notes
                    local_penalty: int = self._cost_function(cf_draft)
                    if local_penalty <= local_max:
                        local_max = local_penalty
                        best_choice = notes
                assert best_choice != -1, "No best choice found"
                cf_shell[i] = best_choice
            self.cf_errors = []
            total_penalty = self._cost_function(cf_shell)
            iteration += 1
        return cf_shell.copy()


class Counterpoint(ABC):
    def __init__(self, cf: CantusFirmus, ctp_position: CtpPositionName = "above", randomizer: random.Random | None = None):
        self.voice_range = RANGES[_shift_part(cf.part, ctp_position)]
        self.melody = Melody(cf.key, cf.scale.scale_type, cf.bar_length, voice_range=self.voice_range)
        self.ctp_position: CtpPositionName = ctp_position
        self.scale_pitches: list[int] = self.melody.scale_pitches
        self.cf: CantusFirmus = cf
        self.search_domain: list[list[int]] = []
        self.ctp_errors: list[str] = []
        self.randomizer = randomizer if randomizer is not None else random.Random()

    @property
    @abstractmethod
    def species(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def ERROR_THRESHOLD(self) -> int:
        return 0

    @abstractmethod
    def get_rhythm(self) -> list[list[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_ties(self) -> list[bool]:
        raise NotImplementedError

    @abstractmethod
    def _possible_notes(self) -> list[list[int]]:
        raise NotImplementedError

    @property
    def MAX_SEARCH_TIME(self) -> int:
        return 5

    def _start_notes(self) -> list[int]:
        """Get all valid start notes"""
        cf_tonic: int = self.cf.pitches[0]
        if self.ctp_position == "above":
            if self.species == 1:
                return [cf_tonic, cf_tonic + P5, cf_tonic + Octave]
            else:
                return [cf_tonic + P5, cf_tonic + Octave]
        else:
            if self.species == 1:
                return [cf_tonic - Octave, cf_tonic]
            else:
                return [cf_tonic - Octave]

    def _end_notes(self) -> list[int]:
        """Get all valid end notes"""
        cf_tonic: int = self.cf.pitches[0]
        if self.ctp_position == "above":
            return [cf_tonic, cf_tonic + Octave]
        else:
            return [cf_tonic, cf_tonic - Octave]

    def _penultimate_notes(self, cf_end: int) -> list[int]:
        """Get all valid penultimate notes"""
        cf_direction: list[float] = [sign(self.cf.pitches[i] - self.cf.pitches[i - 1]) for i in range(1, len(self.cf.pitches))]
        s: int = 1 if self.ctp_position == "above" else -1
        penultimate: int = cf_end + 2 if cf_direction[-1] == 1.0 else cf_end - 1
        return [penultimate, penultimate + s * Octave]

    def get_consonant_possibilities(self, cf_note: int) -> list[int]:
        """Get all consonant possibilities to initialize counterpoint randomly"""
        poss: list[int] = []
        for interval in HARMONIC_CONSONANCES:
            if self.ctp_position == "above":
                if cf_note + interval in self.scale_pitches:
                    poss.append(cf_note + interval)
            else:
                if cf_note - interval in self.scale_pitches:
                    poss.append(cf_note - interval)
        return poss

    def randomize_ctp_melody(self) -> list[int]:
        ctp_melody: list[int] = []
        i: int = 0
        measure: int = 0
        while measure < len(self.melody.rhythm):
            note_duration: int = 0
            while note_duration < len(self.melody.rhythm[measure]):
                if i == 0:
                    ctp_melody.append(self.randomizer.choice(self.search_domain[i]))
                elif i > 0 and self.melody.ties[i - 1]:
                    ctp_melody.append(ctp_melody[i - 1])
                else:
                    ctp_melody.append(self.randomizer.choice(self.search_domain[i]))
                i += 1
                note_duration += 1
            measure += 1
        return ctp_melody

    def generate_ctp(self) -> None:
        self.melody.set_melody(self.randomize_ctp_melody())
        self.ctp_errors = []
        self.error, best_ctp, self.ctp_errors = improved_search(self)
        self.melody.set_melody(best_ctp)


class FirstSpecies(Counterpoint):
    def __init__(self, cf: CantusFirmus, ctp_position: CtpPositionName = "above"):
        super(FirstSpecies, self).__init__(cf, ctp_position)
        self.melody.set_rhythm(self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 50

    @property
    def species(self) -> int:
        return 1

    def get_rhythm(self) -> list[list[int]]:
        "Voices all move together in the same rhythm as the cantus firmus."
        return [[8]] * self.cf.length

    def get_ties(self) -> list[bool]:
        return [False] * self.cf.length

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [[] for _ in self.melody.rhythm]
        for i in range(len(self.melody.rhythm)):
            if i == 0:
                poss[i] = self._start_notes()
            elif i == len(self.melody.rhythm) - 2:
                poss[i] = self._penultimate_notes(self.cf.pitches[i + 1])
            elif i == len(self.melody.rhythm) - 1:
                poss[i] = self._end_notes()
            else:
                poss[i] = self.get_consonant_possibilities(self.cf.pitches[i])
        return poss


class SecondSpecies(Counterpoint):
    def __init__(self, cf: CantusFirmus, ctp_position: CtpPositionName = "above"):
        super(SecondSpecies, self).__init__(cf, ctp_position)
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes: int = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 50

    @property
    def species(self) -> int:
        return 2

    def get_downbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches))) * 2
        return indices[::2]

    def get_upbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches))) * 2
        return indices[1::2]

    def get_rhythm(self) -> list[list[int]]:
        rhythm = [[4, 4]] * (len(self.cf.pitches) - 1)
        rhythm.append([8])
        return rhythm

    def get_ties(self) -> list[bool]:
        return [False] * self.num_notes

    def get_harmonic_possibilities(self, idx: int, cf_note: int) -> list[int]:
        """Get all harmonic possibilities for the counterpoint"""
        poss: list[int] = self.get_consonant_possibilities(cf_note)
        upbeats: list[int] = self.get_upbeats()
        if idx in upbeats:
            if idx != 1:
                for diss in HARMONIC_DISSONANT_INTERVALS:
                    if self.ctp_position == "above":
                        if cf_note + diss in self.scale_pitches:
                            poss.append(cf_note + diss)
                    else:
                        if cf_note - diss in self.scale_pitches:
                            poss.append(cf_note - diss)
        return poss

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [[] for _ in range(self.num_notes)]
        i: int = 0
        for m in range(len(self.get_rhythm())):
            for n in range(len(self.get_rhythm()[m])):
                if m == 0:
                    # First measure. start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.get_rhythm()) - 2 and n == 1:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.get_rhythm()) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(i, self.cf.pitches[m])
                i += 1
        return poss


class ThirdSpecies(Counterpoint):
    def __init__(self, cf: CantusFirmus, ctp_position: CtpPositionName = "above"):
        super(ThirdSpecies, self).__init__(cf, ctp_position)
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes: int = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 100

    @property
    def species(self) -> int:
        return 3

    def get_downbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches)))*4
        return indices[::2]

    def get_upbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches)))*4
        return indices[1::2]

    def get_rhythm(self) -> list[list[int]]:
        rhythm: list[list[int]] = [[2, 2, 2, 2]] * (len(self.cf.pitches) - 1)
        rhythm.append([8])
        return rhythm

    def get_ties(self) -> list[bool]:
        return [False] * self.num_notes

    def get_harmonic_possibilities(self, idx: int, cf_note: int) -> list[int]:
        poss: list[int] = self.get_consonant_possibilities(cf_note)
        upbeats: list[int] = self.get_upbeats()
        if idx in upbeats:
            if idx != 1:
                for diss in HARMONIC_DISSONANT_INTERVALS:
                    if self.ctp_position == "above":
                        if cf_note + diss in self.scale_pitches:
                            poss.append(cf_note + diss)
                    else:
                        if cf_note - diss in self.scale_pitches:
                            poss.append(cf_note - diss)
        return poss

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [[] for _ in range(self.num_notes)]
        i: int = 0
        for m in range(len(self.get_rhythm())):
            for n in range(len(self.get_rhythm()[m])):
                if m == 0 and n in [0, 1]:
                    # First measure, after rest start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.get_rhythm()) - 2 and n == 3:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.get_rhythm()) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(i, self.cf.pitches[m])
                i += 1
        return poss


class FourthSpecies(Counterpoint):
    def __init__(self, cf: CantusFirmus, ctp_position: CtpPositionName = "above"):
        super(FourthSpecies, self).__init__(cf, ctp_position)
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes: int = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 25

    @property
    def species(self) -> int:
        return 4

    def get_rhythm(self) -> list[list[int]]:
        rhythm: list[list[int]] = [[4, 4]] * (self.cf.length - 1)
        rhythm.append([8])
        return rhythm

    def get_ties(self) -> list[bool]:
        ties: list[bool] = []
        for i in range(self.num_notes-2):
            ties.append(i % 2 == 1)
        ties.append(False)
        ties.append(False)
        return ties

    def get_harmonic_possibilities(self, idx: int, cf_note: int) -> list[int]:
        poss: list[int] = self.get_consonant_possibilities(cf_note)
        return poss

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [[] for _ in range(self.num_notes)]
        i: int = 0
        for m in range(len(self.get_rhythm())):
            for n in range(len(self.get_rhythm()[m])):
                if m == 0:
                    # First measure. start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.get_rhythm()) - 2 and n == 1:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.get_rhythm()) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(i, self.cf.pitches[m])
                i += 1
        return poss


class FifthSpecies(Counterpoint):
    def __init__(self, cf: CantusFirmus, ctp_position: CtpPositionName = "above"):
        super(FifthSpecies, self).__init__(cf, ctp_position)
        self.species: str = "fifth"
        self.melody.set_rhythm(self.get_rhythm())
        self.rhythm: list[list[int]] = self.melody.rhythm
        self.num_notes: int = sum(len(row) for row in self.rhythm)
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 100

    """ RHYTHMIC RULES """

    def get_rhythm(self) -> list[list[int]]:
        rhythm: list[list[int]] = []
        measure_rhythms: list[list[int]] = [
            [2, 2, 2, 2],
            [4, 2, 2],
            [2, 2, 4],
            [4, 4],
            [2, 1, 1, 2, 2],
            [2, 1, 1, 4],
            [4, 2, 1, 1],
            [2, 2, 2, 1, 1],
            [2, 1, 1, 2, 2]
        ]
        rhythmic_weights: list[int] = [75, 75, 75, 75, 10, 5, 5, 5, 5]
        for measures in range(len(self.cf.pitches)-1):
            if measures == 0:
                rhythm.append([4, 4])
            else:
                rhythm.append(self.randomizer.choices(measure_rhythms, rhythmic_weights)[0])
        rhythm.append([8,])
        return rhythm

    def get_ties(self) -> list[bool]:
        rhythm: list[list[int]] = self.rhythm
        ties: list[bool] = []
        for m in range(len(rhythm)-1):
            for n in range(len(rhythm[m])):
                if m == 0 and n == 1:
                    ties.append(True)
                elif m > 0 and n == len(rhythm[m])-1:
                    ties.append(rhythm[m+1][0] == rhythm[m][n]/2)
                else:
                    ties.append(False)
        ties.append(False)
        ties.append(False)
        return ties

    def get_harmonic_possibilities(self, m: int, n: int, cf_note: int) -> list[int]:
        add_dissonance = self.rhythm[m][n] == 1 or sum(self.rhythm[m][:n]) in [2, 6]
        poss: list[int] = self.get_consonant_possibilities(cf_note)
        if add_dissonance:
            for diss in HARMONIC_DISSONANT_INTERVALS:
                if self.ctp_position == "above":
                    if cf_note + diss in self.scale_pitches:
                        poss.append(cf_note + diss)
                else:
                    if cf_note - diss in self.scale_pitches:
                        poss.append(cf_note - diss)
        return poss

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [[] for _ in range(self.num_notes)]
        i: int = 0
        for m in range(len(self.rhythm)):
            for n in range(len(self.rhythm[m])):
                if m == 0:
                    # First measure. start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.rhythm) - 2 and n == len(self.rhythm[m])-1:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.rhythm) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(m, n, self.cf.pitches[m])
                i += 1
        return poss


def _get_indices(ctp_len: int, idx: int, n_window: int) -> list[int]:
    s_w: list[int] = []
    for i in range(n_window):
        if idx + i < ctp_len:
            s_w.append(idx + i)
        else:
            s_w.append(ctp_len - 1 - i)
    s_w.sort()
    return [s_w[0], s_w[-1]]


def _path_search(ctp: Counterpoint, search_window: list[int]) -> tuple[list[int], float, list[str]]:
    raise NotImplementedError


def search(ctp: Counterpoint) -> tuple[float, list[int], list[str]]:
    raise NotImplementedError


def brute_force(ctp: Counterpoint) -> tuple[float, list[int], list[str]]:
    raise NotImplementedError


def best_first_search(ctp: Counterpoint, weighted_idx: list[int]):
    search_domain = ctp.search_domain
    search_ctp = ctp.melody.pitches
    best_global_ctp = search_ctp.copy()
    best_global_error = INFINITY
    best_global_weighted_indices: dict[int, int] = {}
    if isinstance(weighted_idx, list):
        idx = weighted_idx
    else:
        idx = list(weighted_idx.keys())
    for i in idx:
        best_note = search_domain[i][0]
        local_error = INFINITY
        local_weighted_indices = {}
        for j in range(len(search_domain[i])):
            search_ctp[i] = search_domain[i][j]
            ctp.melody.set_melody(search_ctp.copy())
            constrained = Constraints(ctp)
            error = constrained.cost_function()
            weighted_indices = constrained.get_weighted_indices()
            if error <= local_error:
                best_note = search_domain[i][j]
                local_error = error
                local_weighted_indices = weighted_indices
        search_ctp[i] = best_note
        if local_error < best_global_error:
            best_global_ctp = search_ctp.copy()
            best_global_error = local_error
            best_global_weighted_indices = local_weighted_indices
    return best_global_error, best_global_ctp, best_global_weighted_indices


def improved_search(ctp: Counterpoint):
    """The main search function for the counterpoint generation - returns
    - the lowest penalty
    - the best counterpoint
    - the list of errors"""
    start_time = time.time()
    penalty = INFINITY
    elapsed_time = 0.
    best_ctp = ctp.melody.pitches
    lowest_penalty = INFINITY
    weighted_idx = [i for i in range(len(best_ctp))]
    prev_penalty = penalty
    randomize_idx = 1
    while penalty >= ctp.ERROR_THRESHOLD and elapsed_time < ctp.MAX_SEARCH_TIME:
        penalty, ctp_notes, weighted_idx = best_first_search(ctp, weighted_idx)
        if penalty == prev_penalty:  # no improvement
            weighted_idx = list(weighted_idx.keys())
            for i in range(randomize_idx):
                ctp_notes[weighted_idx[i]] = ctp.randomizer.choice(ctp.search_domain[weighted_idx[i]])
            ctp.randomizer.shuffle(weighted_idx)
            ctp.melody.set_melody(ctp_notes)
            if randomize_idx != len(best_ctp)-1:
                randomize_idx += 1
        if penalty < lowest_penalty:
            randomize_idx = 1
            best_ctp = ctp_notes
            ctp.melody.set_melody(best_ctp)
            lowest_penalty = penalty
            weighted_idx = weighted_idx
        elapsed_time = time.time() - start_time
        prev_penalty = penalty
    constraint = Constraints(ctp)
    lowest_penalty = constraint.cost_function()
    lowest_error_list = constraint.get_errors()
    return lowest_penalty, best_ctp, lowest_error_list


class CounterpointGenerator(SongGenerator):
    def __init__(
        self,
        key: KeyName,
        scale_name: ScaleName,
        species: SpeciesName,
        cf_part: PartName,
        ctp_position: CtpPositionName | None = None,
        bar_length: int = 2,
        seed: int | None = None
    ):
        super(CounterpointGenerator, self).__init__(seed)
        self.key: KeyName = key
        self.scale_name: ScaleName = scale_name
        self.species: SpeciesName = species
        self.cf_part: PartName = cf_part
        if cf_part == "bass":
            self.ctp_position: CtpPositionName = "above"
        elif cf_part == "soprano":
            self.ctp_position: CtpPositionName = "below"
        else:
            self.ctp_position = "above" if ctp_position is None else ctp_position
        self.bar_length: int = bar_length

    def generate(self) -> list[Note]:
        cf = CantusFirmus(
            self.key,
            self.scale_name,
            self.bar_length,
            part=self.cf_part,
            randomizer=self.randomizer
        )
        ctp: Counterpoint = {
            "first": FirstSpecies,
            "second": SecondSpecies,
            "third": ThirdSpecies,
            "fourth": FourthSpecies,
            "fifth": FifthSpecies
        }[self.species](cf, ctp_position=self.ctp_position, randomizer=self.randomizer)
        ctp.generate_ctp()
        notes = cf.to_list_notes() + ctp.melody.to_list_notes()
        return notes
