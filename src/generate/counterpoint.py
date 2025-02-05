from __future__ import annotations
import matplotlib.pyplot as plt
import json
import itertools
import math
import random as rm
import time
import typing
from ..data import PIANO_A0, PIANO_C8, Note
from .constants import *
from .constraints import Constraints

ScaleName = typing.Literal["major", "minor"]

INFINITY = 1 << 64 - 1  # A big number for the search algorithm


def _make_note(pitch: int | str) -> Note:
    if isinstance(pitch, int):
        return Note.from_midi_number(pitch)
    elif isinstance(pitch, str):
        return Note.from_str(pitch)
    else:
        raise ValueError("Invalid pitch type")


class Scale:
    """
    The scale class manages scale object. This include the construction of scales included in the NAMED_SCALE dict.
    the scale is a list of all possible notes in the given scale across the entire piano. This means that the root note
    is not necessarily the lowest note.
    """

    def __init__(self, key: str, scale: str | Scale, scale_range: list[int] | None = None):
        if key[0].upper() not in (KEY_NAMES_SHARP or KEY_NAMES):
            raise ValueError(f"Invalid key name: {key}")

        # Get the lowest octave
        if key in ["A", "A#", "B", "Bb"]:
            oct = 0
        else:
            oct = 1
        self.root: Note = _make_note(key + str(oct))  # sets the root of the scale in valid string format
        self.key: str = key
        self.scale_type: str | Scale = scale
        if isinstance(scale, str):
            self.intervals = Scale.intervals_from_name(scale)
        elif isinstance(scale, Scale):
            self.intervals = scale.intervals
        else:
            raise ValueError("Invalid scale type")
        self.scale: list[Note] = self.build_scale()
        self.scale_pitches: list[int] = self.get_scale_pitches()
        if scale_range is not None:
            self.limit_range(scale_range)

    @staticmethod
    def intervals_from_name(scale_name: str) -> tuple[int, ...]:
        global NAMED_SCALES
        scale_name = scale_name.lower()

        # supporting alternative formatting..
        for text in ['scale', 'mode']:
            scale_name = scale_name.replace(text, '')
        for text in [" ", "-"]:
            scale_name = scale_name.replace(text, "_")
        return tuple(NAMED_SCALES[scale_name])

    def build_scale(self) -> list[Note]:
        start_pitch = self.root.midi_number
        scale_len = len(self.intervals)
        highest_possible_pitch = PIANO_C8
        lowest_possible_pitch = PIANO_A0
        j = 0
        scale: list[Note] = []
        pitch = start_pitch
        # adds all possible values above the root pitch
        while pitch <= highest_possible_pitch:
            scale.append(_make_note(pitch))
            pitch = scale[j].midi_number + self.intervals[j % scale_len]
            j += 1
        # adds all possible values under the root pitch
        j = scale_len - 1
        pitch = start_pitch - self.intervals[j % scale_len]
        while pitch >= lowest_possible_pitch:
            scale.insert(0, _make_note(pitch))
            j -= 1
            pitch = pitch - self.intervals[j % scale_len]
        return scale

    def get_scale_pitches(self) -> list[int]:
        scale_pitches: list[int] = []
        for notes in self.scale:
            scale_pitches.append(notes.midi_number)
        return scale_pitches

    def get_scale_range(self, scale_range: list[int]) -> list[int]:
        """
        :param scale_range: [int] list of note pitches in the range to be returned
        :return: the scale limited to the given range
        """
        scale_pitches: list[int] = []
        for notes in scale_range:
            if notes in self.scale_pitches:
                scale_pitches.append(notes)
        return scale_pitches

    def limit_range(self, scale_range: list[int]) -> None:
        scale: list[Note] = []
        for notes in scale_range:
            if notes in self.scale_pitches:
                scale.append(_make_note(notes))
        self.scale = scale

    def set_time(self, duration: float) -> None:
        raise NotImplementedError


class Melody:
    def __init__(
            self,
            key: str,
            scale: str,
            bar_length: float,
            melody_notes: list[int] | None = None,
            melody_rhythm: list[float] | None = None,
            ties: list[bool] | None = None,
            start: int = 0,
            voice_range: list[int] | None = None
    ):
        self.key: str = key
        self.scale_name: str = scale
        self.voice_range: list[int] | None = voice_range
        self.scale: Scale = Scale(key, scale, voice_range)
        self.scale_pitches: list[int] = self.scale.get_scale_pitches()
        self.note_resolution: int = 8
        self.start: int = start
        self.bar_length: float = float(bar_length)

        self.pitches: list[int] | None = melody_notes
        self.rhythm: list[float] | None = melody_rhythm
        self.ties: list[bool] | None = ties
        if self.pitches is not None:
            self.search_domain: list[list[int]] = [self.scale_pitches for notes in self.pitches]
        else:
            self.search_domain: list[list[int]] = [self.scale_pitches]

    def set_ties(self, ties: list[bool]) -> None:
        self.ties = ties.copy()

    def set_rhythm(self, rhythm: list[float]) -> None:
        self.rhythm = rhythm.copy()

    def set_melody(self, melody: list[int]) -> None:
        self.pitches = melody.copy()

    def get_ties(self) -> list[bool]:
        return self.ties.copy() if self.ties is not None else []

    def get_rhythm(self) -> list[float]:
        return self.rhythm.copy() if self.rhythm is not None else []

    def get_melody(self) -> list[int]:
        return self.pitches.copy() if self.pitches is not None else []

    def get_end_time(self) -> float:
        t = self.start
        if self.rhythm is not None:
            for elem in self.rhythm:
                t += elem
        return t * self.bar_length / float(self.note_resolution)


class CantusFirmus(Melody):
    # Some constants for easy access
    perfect_intervals: list[int] = [Unison, P5, Octave]
    dissonant_intervals: list[int] = [m7, M7, Tritone, -m6, -m7, -M7]
    consonant_melodic_intervals: list[int] = [m2, M2, m3, M3, P4, P5, m6, Octave, -m2, -M2, -m3, -M3, -P4, -P5, -Octave]

    def __init__(
            self,
            key: str,
            scale: str,
            bar_length: float,
            melody_notes: list[int] | None = None,
            melody_rhythm: list[float] | None = None,
            start: int = 0,
            voice_range: list[int] = RANGES[ALTO]
    ):
        super(CantusFirmus, self).__init__(
            key, scale, bar_length,
            melody_notes=melody_notes,
            melody_rhythm=melody_rhythm,
            start=start,
            voice_range=voice_range
        )
        self.cf_errors: list[str] = []
        self.rhythm: list[tuple[int]] = self._generate_rhythm()
        self.ties: list[bool] = [False]*len(self.rhythm)
        self.pitches: list[int] = self._generate_cf()
        self.length: int = len(self.rhythm)

    def _start_note(self) -> tuple[list[int], int]:
        root: str = self.key
        try:
            root_idx: int = KEY_NAMES.index(root)
        except:
            root_idx = KEY_NAMES_SHARP.index(root)
        v_range: list[int] = self.voice_range
        possible_start_notes: list[int] = []
        for pitches in v_range:
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
        penultimate_note: int = rm.choices([leading_tone, super_tonic], weights)[0]
        return penultimate_note

    def _get_leading_tones(self) -> list[int]:
        if self.scale_name == "minor":
            leading_tone: int = self._start_note()[1] - 2
        else:
            leading_tone = self._start_note()[1] - 1
        return [leading_tone]

    def _generate_rhythm(self) -> list[tuple[int]]:
        """
        Generates a random rhythm for the cantus firmus
        Empirically, 12 seems to be the most common, but the rhythm can be any length between 8 and 14
        """
        random_length: int = rm.choice(list(range(8, 15)) + [12] * 2)
        return [(8,)] * random_length

    def _is_step(self, note: int, prev_note: int) -> bool:
        if abs(prev_note-note) in [m2, M2]:
            return True
        else:
            return False

    def _is_small_leap(self, note: int, prev_note: int) -> bool:
        if abs(prev_note-note) in [m3, M3]:
            return True
        else:
            return False

    def _is_large_leap(self, note: int, prev_note: int) -> bool:
        if abs(prev_note-note) >= P4:
            return True
        else:
            return False

    def _is_climax(self, cf_shell: list[int]) -> bool:
        if cf_shell.count(max(cf_shell)) == 1:
            return True
        else:
            return False

    def _is_resolved_leading_tone(self, cf_shell: list[int]) -> bool:
        tonics: list[int] = self._start_note()[0]
        leading_tones: list[int] = self._get_leading_tones()
        for leading_tone in leading_tones:
            if leading_tone in cf_shell and cf_shell[cf_shell.index(leading_tone)+1] != tonics[0]:
                return False
        return True

    def _is_dissonant_intervals(self, cf_shell: list[int]) -> bool:
        for i in range(len(cf_shell)-1):
            if cf_shell[i+1] - cf_shell[i] in self.dissonant_intervals:
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
        for notes in set(cf_shell):
            if cf_shell.count(notes) > 4:
                return False
            else:
                return True

    def _is_valid_range(self, cf_shell: list[int]) -> bool:
        if abs(max(cf_shell)-min(cf_shell)) > Octave+M3:
            return False
        else:
            return True

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
        cf_shell: list[int] = [rm.choice(self.scale_pitches) for i in range(length)]
        cf_shell[0] = start_note
        cf_shell[-1] = end_note
        cf_shell[-2] = penultimate_note
        return cf_shell

    def _get_melodic_consonances(self, prev_note: int) -> list[int]:
        mel_cons: list[int] = []
        for intervals in self.consonant_melodic_intervals:
            if prev_note+intervals in self.scale_pitches:
                mel_cons.append(prev_note+intervals)
        # To further randomize the generated results, the melodic consonances are shuffled
        rm.shuffle(mel_cons)
        return mel_cons

    def _generate_cf(self) -> list[int]:
        total_penalty: int = INFINITY
        iteration: int = 0
        while total_penalty > 0:
            cf_shell: list[int] = self._initialize_cf()  # initialized randomly
            for i in range(1, len(cf_shell)-2):
                self.cf_errors = []
                local_max: int = INFINITY
                cf_draft: list[int] = cf_shell.copy()
                mel_cons: list[int] = self._get_melodic_consonances(cf_shell[i-1])
                for notes in mel_cons:
                    cf_draft[i] = notes
                    local_penalty: int = self._cost_function(cf_draft)
                    if local_penalty <= local_max:
                        local_max = local_penalty
                        best_choice: int = notes
                cf_shell[i] = best_choice
            self.cf_errors = []
            total_penalty = self._cost_function(cf_shell)
            iteration += 1
        return cf_shell.copy()


class Counterpoint:
    def __init__(self, cf: CantusFirmus, ctp_position: str = "above"):
        if ctp_position == "above":
            self.voice_range: list[int] = RANGES[RANGES.index(cf.voice_range) + 1]
        else:
            self.voice_range: list[int] = RANGES[RANGES.index(cf.voice_range) - 1]
        self.melody: Melody = Melody(cf.key, cf.scale, cf.bar_length, voice_range=self.voice_range)
        self.ctp_position: str = ctp_position
        self.scale_pitches: list[int] = self.melody.scale_pitches
        self.cf: CantusFirmus = cf
        self.species: str | None = None
        self.search_domain: list[list[int]] = []
        self.ctp_errors: list[str] = []

    @property
    def MAX_SEARCH_TIME(self) -> int:
        return 5

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 0

    """ VALID START, END, AND PENULTIMATE NOTES"""

    def _start_notes(self) -> list[int]:
        cf_tonic: int = self.cf.pitches[0]
        if self.ctp_position == "above":
            if SPECIES[self.species] == 1:
                return [cf_tonic, cf_tonic + P5, cf_tonic + Octave]
            else:
                return [cf_tonic + P5, cf_tonic + Octave]
        else:
            if SPECIES[self.species] == 1:
                return [cf_tonic - Octave, cf_tonic]
            else:
                return [cf_tonic - Octave]

    def _end_notes(self) -> list[int]:
        cf_tonic: int = self.cf.pitches[0]
        if self.ctp_position == "above":
            return [cf_tonic, cf_tonic + Octave]
        else:
            return [cf_tonic, cf_tonic - Octave]

    def _penultimate_notes(self, cf_end: int) -> list[int]:
        cf_direction: list[int] = [sign(self.cf.pitches[i] - self.cf.pitches[i - 1]) for i in range(1, len(self.cf.pitches))]
        if self.ctp_position == "above":
            s: int = 1
        else:
            s: int = -1
        if cf_direction[-1] == 1.0:
            penultimate: int = cf_end + 2
        else:
            penultimate: int = cf_end - 1
        return [penultimate, penultimate + s * Octave]

    """ INITIALIZING COUNTERPOINT WITH RANDOM VALUES"""

    def get_consonant_possibilities(self, cf_note: int) -> list[int]:
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
                    ctp_melody.append(rm.choice(self.search_domain[i]))
                elif i > 0 and self.melody.ties[i - 1]:
                    ctp_melody.append(ctp_melody[i - 1])
                else:
                    ctp_melody.append(rm.choice(self.search_domain[i]))
                i += 1
                note_duration += 1
            measure += 1
        return ctp_melody

    """ GENERATE COUNTERPOINT PITCHES BY CALLING THE SEARCH ALGORITHM"""

    def generate_ctp(self) -> None:
        if self.species is None:
            print("No species to generate!")
        self.melody.set_melody(self.randomize_ctp_melody())
        self.ctp_errors = []
        self.error, best_ctp, self.ctp_errors = search.improved_search(self)
        self.melody.set_melody(best_ctp)


class FirstSpecies(Counterpoint):
    def __init__(self, cf: CantusFirmus, ctp_position: str = "above"):
        super(FirstSpecies, self).__init__(cf, ctp_position)
        self.species: str = "first"
        self.melody.set_rhythm(self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 50

    """ RHYTHM """

    def get_rhythm(self) -> list[tuple[int]]:
        "Voices all move together in the same rhythm as the cantus firmus."
        return [(8,)] * self.cf.length

    def get_ties(self) -> list[bool]:
        return [False] * self.cf.length

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [None for _ in self.melody.rhythm]
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
    def __init__(self, cf: CantusFirmus, ctp_position: str = "above"):
        super(SecondSpecies, self).__init__(cf, ctp_position)
        self.species: str = "second"
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes: int = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 50

    """ HELP FUNCTIONS"""

    def get_downbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches))) * 2
        return indices[::2]

    def get_upbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches))) * 2
        return indices[1::2]

    """ RHYTHMIC RULES """

    def get_rhythm(self) -> list[tuple[int]]:
        rhythm: list[tuple[int]] = [(4, 4)] * (len(self.cf.pitches) - 1)
        rhythm.append((8,))
        return rhythm

    def get_ties(self) -> list[bool]:
        return [False] * self.num_notes

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, idx: int, cf_note: int) -> list[int]:
        poss: list[int] = super(SecondSpecies, self).get_consonant_possibilities(cf_note)
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
        poss: list[list[int]] = [None for _ in range(self.num_notes)]
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
    def __init__(self, cf: CantusFirmus, ctp_position: str = "above"):
        super(ThirdSpecies, self).__init__(cf, ctp_position)
        self.species: str = "third"
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes: int = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 100

    """ HELP FUNCTIONS"""

    def get_downbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches)))*4
        return indices[::2]

    def get_upbeats(self) -> list[int]:
        indices: list[int] = list(range(len(self.cf.pitches)))*4
        return indices[1::2]

    """ RHYTHMIC RULES """

    def get_rhythm(self) -> list[tuple[int]]:
        rhythm: list[tuple[int]] = [(2, 2, 2, 2)] * (len(self.cf.pitches) - 1)
        rhythm.append((8,))
        return rhythm

    def get_ties(self) -> list[bool]:
        return [False] * self.num_notes

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, idx: int, cf_note: int) -> list[int]:
        poss: list[int] = super(ThirdSpecies, self).get_consonant_possibilities(cf_note)
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
        poss: list[list[int]] = [None for _ in range(self.num_notes)]
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
    def __init__(self, cf: CantusFirmus, ctp_position: str = "above"):
        super(FourthSpecies, self).__init__(cf, ctp_position)
        self.species: str = "fourth"
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes: int = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 25

    """ HELP FUNCTIONS"""

    """ RHYTHMIC RULES """

    def get_rhythm(self) -> list[tuple[int]]:
        rhythm: list[tuple[int]] = [(4, 4)] * (self.cf.length - 1)
        rhythm.append((8,))
        return rhythm

    def get_ties(self) -> list[bool]:
        ties: list[bool] = []
        for i in range(self.num_notes-2):
            if i % 2 == 0:
                ties.append(False)
            else:
                ties.append(True)
        ties.append(False)
        ties.append(False)
        return ties

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, idx: int, cf_note: int) -> list[int]:
        poss: list[int] = super(FourthSpecies, self).get_consonant_possibilities(cf_note)
        return poss

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [None for _ in range(self.num_notes)]
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
    def __init__(self, cf: CantusFirmus, ctp_position: str = "above"):
        super(FifthSpecies, self).__init__(cf, ctp_position)
        self.species: str = "fifth"
        self.melody.set_rhythm(self.get_rhythm())
        self.rhythm: list[tuple[int]] = self.melody.get_rhythm()
        self.num_notes: int = sum(len(row) for row in self.rhythm)
        self.melody.set_ties(self.get_ties())
        self.search_domain: list[list[int]] = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    @property
    def ERROR_THRESHOLD(self) -> int:
        return 100

    """ RHYTHMIC RULES """

    def get_rhythm(self) -> list[tuple[int]]:
        rhythm: list[tuple[int]] = []
        measure_rhythms: list[tuple[int]] = [(2, 2, 2, 2), (4, 2, 2), (2, 2, 4), (4, 4),
                                             (2, 1, 1, 2, 2), (2, 1, 1, 4), (4, 2, 1, 1), (2, 2, 2, 1, 1), (2, 1, 1, 2, 2)]
        rhythmic_weights: list[int] = [75, 75, 75, 75, 10, 5, 5, 5, 5]
        for measures in range(len(self.cf.pitches)-1):
            if measures == 0:
                rhythm.append((4, 4))
            else:
                rhythm.append(rm.choices(measure_rhythms, rhythmic_weights)[0])
        rhythm.append((8,))
        return rhythm

    def get_ties(self) -> list[bool]:
        rhythm: list[tuple[int]] = self.rhythm
        ties: list[bool] = []
        for m in range(len(rhythm)-1):
            for n in range(len(rhythm[m])):
                if m == 0 and n == 1:
                    ties.append(True)
                elif m > 0 and n == len(rhythm[m])-1:
                    if rhythm[m+1][0] == rhythm[m][n]/2:
                        ties.append(True)
                    else:
                        ties.append(False)
                else:
                    ties.append(False)
        ties.append(False)
        ties.append(False)
        return ties

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, m: int, n: int, cf_note: int) -> list[int]:
        add_diss: bool = False
        if self.rhythm[m][n] == 1:
            add_diss = True
        if sum(self.rhythm[m][:n]) in [2, 6]:
            add_diss = True
        poss: list[int] = super(FifthSpecies, self).get_consonant_possibilities(cf_note)
        if add_diss:
            for diss in HARMONIC_DISSONANT_INTERVALS:
                if self.ctp_position == "above":
                    if cf_note + diss in self.scale_pitches:
                        poss.append(cf_note + diss)
                else:
                    if cf_note - diss in self.scale_pitches:
                        poss.append(cf_note - diss)
        return poss

    def _possible_notes(self) -> list[list[int]]:
        poss: list[list[int]] = [None for _ in range(self.num_notes)]
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
    paths: list[list[int]] = []
    ctp_draft: list[int] = ctp.melody.get_melody().copy()
    poss: list[list[int]] = ctp.search_domain
    for i in itertools.product(*poss[search_window[0]:search_window[1] + 1]):
        paths.append(list(i))
    ctp_draft[search_window[0]:search_window[1] + 1] = paths[0]
    best_ctp: list[int] = ctp_draft.copy()
    best_local_error: float = math.inf
    best_error_list: list[str] = []
    for path in paths:
        ctp_draft[search_window[0]:search_window[1] + 1] = path
        ctp.melody.set_melody(ctp_draft)
        enforced_constraints = Constraints(ctp)
        local_error: float = enforced_constraints.get_penalty()
        ctp_errors: list[str] = enforced_constraints.get_errors()
        weighted_indices: list[int] = enforced_constraints.get_weighted_indices()
        if local_error < best_local_error:
            best_ctp = ctp_draft.copy()
            ctp.melody.set_melody(best_ctp)
            best_local_error = local_error
            best_error_list = ctp_errors
    return best_ctp.copy(), best_local_error, best_error_list


def search(ctp: Counterpoint) -> tuple[float, list[int], list[str]]:
    error: float = math.inf
    best_scan_error: float = math.inf
    j: int = 1
    best_ctp: list[int] = ctp.melody.get_melody().copy()
    best_error_list: list[str] = []
    while error >= ctp.ERROR_THRESHOLD and j <= ctp.MAX_SEARCH_WIDTH:
        error_window: float = math.inf
        for i in range(len(ctp.melody.get_melody())):
            window_n: list[int] = _get_indices(len(ctp.melody.get_melody()), i, 2)
            ctp_draft, error, list_of_errors = _path_search(ctp, window_n)
            if i == 0:
                error_window = error
            if error < best_scan_error:
                best_ctp = ctp_draft.copy()
                ctp.melody.set_melody(best_ctp)
                best_scan_error = error
                best_error_list = list_of_errors
                if error < ctp.ERROR_THRESHOLD:
                    return best_scan_error, best_ctp, best_error_list
        ctp.melody.set_melody(best_ctp)
        if error_window >= best_scan_error:
            j += 1
    return best_scan_error, best_ctp, best_error_list


def brute_force(ctp: Counterpoint) -> tuple[float, list[int], list[str]]:
    penalty: float = math.inf
    errors: list[str] = []
    while penalty > ctp.ERROR_THRESHOLD:
        ctp.melody.set_melody(ctp.randomize_ctp_melody())
        enforced_constraints = Constraints(ctp)
        local_error: float = enforced_constraints.get_penalty()
        ctp_errors: list[str] = enforced_constraints.get_errors()
        penalty = local_error
        errors = ctp_errors
    return penalty, ctp.melody.get_melody(), errors


def best_first_search(ctp: Counterpoint, weighted_idx: list[int] | dict[int, float]) -> tuple[float, list[int], list[int]]:
    search_domain: list[list[int]] = ctp.search_domain
    search_ctp: list[int] = ctp.melody.get_melody()
    best_global_ctp: list[int] = search_ctp.copy()
    best_global_error: float = math.inf
    best_global_weighted_indices: list[int] = []
    if isinstance(weighted_idx, list):
        idx: list[int] = weighted_idx
    else:
        idx = list(weighted_idx.keys())
    for i in idx:
        best_note: int = search_domain[i][0]
        local_error: float = math.inf
        local_weighted_indices: list[int] = []
        for j in range(len(search_domain[i])):
            search_ctp[i] = search_domain[i][j]
            ctp.melody.set_melody(search_ctp.copy())
            constrained = Constraints(ctp)
            error: float = constrained.cost_function()
            weighted_indices: list[int] = constrained.get_weighted_indices()
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


def improved_search(ctp: Counterpoint) -> tuple[float, list[int], list[str]]:
    start_time: float = time.time()
    penalty: float = math.inf
    elapsed_time: float = time.time() - start_time
    best_ctp: list[int] = ctp.melody.get_melody()
    lowest_penalty: float = math.inf
    weighted_idx: list[int] = [i for i in range(len(best_ctp))]
    prev_penalty: float = penalty
    randomize_idx: int = 1
    while penalty >= ctp.ERROR_THRESHOLD and elapsed_time < ctp.MAX_SEARCH_TIME:
        penalty, ctp_notes, weighted_idx = best_first_search(ctp, weighted_idx)
        if penalty == prev_penalty:  # no improvement
            weighted_idx = list(weighted_idx.keys())
            for i in range(randomize_idx):
                ctp_notes[weighted_idx[i]] = rm.choice(ctp.search_domain[weighted_idx[i]])
            rm.shuffle(weighted_idx)
            ctp.melody.set_melody(ctp_notes)
            if randomize_idx != len(best_ctp) - 1:
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
    lowest_error_list: list[str] = constraint.get_errors()
    return lowest_penalty, best_ctp, lowest_error_list


def main():
    cont = True
    i = 0
    print("Automatic Species Generation")
    while cont:
        print(" ")
        key = input("key? :")
        scale_name = input("scale type? [major or minor]: ")
        species = input("Species? [first to fifth]: ")
        range_str = input("Voice range of cantus firmus? [bass, tenor, alto, soprano]: ")
        cf_range = RANGES[RANGES_NAMES[range_str]]
        if cf_range == BASS_RANGE:
            ctp_position = "y"
        elif cf_range == SOPRANO_RANGE:
            ctp_position = "n"
        else:
            ctp_position = input("above cantus firmus? [y/n]: ")
        if ctp_position[0].upper() == "Y":
            ctp_position = "above"
        else:
            ctp_position = "below"
        instrument = input("instrument? [Acoustic Grand Piano, Church Organ etc.]: ")
        name = "ctp"+str(i)
        mid_gen = MidiGenerator(key, scale_name, species, ctp_position=ctp_position, cf_range=cf_range)
        mid_gen.set_instrument(instrument)
        mid_gen.to_instrument()
        mid_gen.export_to_midi(name="generated_midi/user_defined/"+species+"_species_"+name+".mid")
        print("midi successfully exported to "+"generated_midi/user_defined/"+species+"_species_"+name+".mid")
        cont_str = input("try again? [y/n]: ")
        if cont_str[0].upper() == "Y":
            cont = True
        else:
            cont = False
        i += 1


class MidiGenerator:
    instruments = ["Church Organ", "Church Organ"]

    def __init__(self, key, scale_name, species, bar_length=2, ctp_position="above", cf_range=RANGES[TENOR], cf_notes=None, cf_rhythm=None):
        self.cf_range_name = RANGES.index(cf_range)
        self.species = species
        self.cf = CantusFirmus(key, scale_name, bar_length, cf_notes, cf_rhythm, start=0, voice_range=cf_range)
        self.loaded_instruments = []
        if species == "first":
            self.ctp = FirstSpecies(self.cf, ctp_position=ctp_position)
        elif species == "second":
            self.ctp = SecondSpecies(self.cf, ctp_position=ctp_position)
        elif species == "third":
            self.ctp = ThirdSpecies(self.cf, ctp_position=ctp_position)
        elif species == "fourth":
            self.ctp = FourthSpecies(self.cf, ctp_position=ctp_position)
        elif species == "fifth":
            self.ctp = FifthSpecies(self.cf, ctp_position=ctp_position)
        self.ctp.generate_ctp()
        print(self.ctp.search_domain)

    def set_instrument(self, name):
        if isinstance(name, list):
            self.instruments = name
        else:
            self.instruments = [name]*2

    def to_instrument(self):
        cf_inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(self.instruments[0]), name="cf")
        ctp_inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(self.instruments[1]), name="ctp")
        self.ctp.melody.to_instrument(ctp_inst)
        self.loaded_instruments.append(ctp_inst)
        self.cf.to_instrument(cf_inst)
        self.loaded_instruments.append(cf_inst)

    def export_to_midi(self, tempo=120, name="generated_midi/user_defined/ctp.mid"):
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        for inst in self.loaded_instruments:
            if inst != None:
                pm.instruments.append(inst)
        pm.write(name)


def result_generation():
    inst = ["Acoustic Grand Piano"] * 4
    data = []
    for i in range(100):
        iteration_data = {}
        key = rm.choice(KEY_NAMES)
        scale = rm.choice(["major", "minor"])
        cf_range = rm.choice([BASS, TENOR, ALTO, SOPRANO])
        if cf_range == SOPRANO:
            ctp_position = "below"
        elif cf_range == BASS:
            ctp_position = "above"
        else:
            ctp_position = rm.choice(["above", "below"])
        start = time()
        ctp = MidiGenerator(key, scale, ctp_position=ctp_position, cf_range=RANGES[cf_range], bar_length=2)
        end = time()-start
        iteration_data["index"] = i
        iteration_data["error"] = ctp.ctp.error
        iteration_data["time"] = end
        iteration_data["penalties"] = ctp.penalty_list
        iteration_data["error_list"] = ctp.ctp.ctp_errors
        ctp.set_instrument(inst)
        ctp.to_instrument()
        ctp.export_to_midi(tempo=120, name="generated_midi/fifth_species/data2" + str(i) + ".mid")
        data.append(iteration_data)
    with open('data/fifth_species/data2.txt', 'w') as filehandle:
        json.dump(data, filehandle)
# result_generation()


def result_analysis():
    with open('data/fifth_species/data.txt', 'r') as filehandle:
        data = json.load(filehandle)
    THRESHOLD = 100
    errors = []
    penalties = []
    errors_list = []
    time_list = []
    zero_errors = []
    for i in range(len(data)):
        errors.append(data[i]["error"])
        penalties.append(data[i]["penalties"])
        errors_list.append(data[i]["error_list"])
        if errors[i] == 0:
            zero_errors.append(i)
        time_list.append(data[i]["time"])
    print("average time: ", sum(time_list)/len(data))
    print("average error: ", sum(errors)/len(data))
    print("worst case :", max(errors), "at index ", errors.index(max(errors)), "with errors ", errors_list[errors.index(max(errors))], "and comp time ", time_list[errors.index(max(errors))])
    print("zero errors: ", zero_errors)
    most_common_error = {}
    for i in errors_list:
        for e in i:
            if e not in most_common_error.keys():
                most_common_error[e] = 1
            else:
                most_common_error[e] += 1
    most_common_error = dict(sorted(most_common_error.items(), reverse=True, key=lambda item: item[1]))
    most_common_error_list = list(most_common_error.keys())
    num_errors = 0
    for key in most_common_error:
        num_errors += most_common_error[key]
    print("most common error: ", most_common_error_list[0])
    print("second most common error: ", most_common_error_list[1])
    print(f"precentage: ", (most_common_error[most_common_error_list[0]]/num_errors)*100)
    num_below_threshold = 0
    for e in errors:
        if e < THRESHOLD:
            num_below_threshold += 1
    print("precentage below threshold: ", num_below_threshold, "%")
    print("penalties: ", penalties[8])
    plt.plot(penalties[8])
    plt.ylabel('penalty')
    plt.xlabel("iteration nr")
    plt.show()
# result_analysis()
