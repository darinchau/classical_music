from __future__ import annotations
import os
import copy
import mido
import music21 as m21
import re
from dataclasses import dataclass
from functools import reduce
from mido import MidiFile, MidiTrack, Message
from typing import Literal
from ..util import is_ipython, PIANO_A0, PIANO_C8
from .base import SymbolicMusic
import typing

_PITCH_NAME_REGEX = re.compile(r"([CDEFGAB])(#+|b+)?(-?[0-9]+)")


@dataclass(frozen=True)
class Note:
    """A piano note is a representation of a note on the piano, with a note name and an octave
    The convention being middle C is C4. The lowest note is A0 and the highest note is C8.

    If the note is in real time, then the duration and offset is timed with respect to quarter length,
    otherwise it is timed with respect to real-time seconds."""
    index: int
    octave: int
    duration: float
    offset: float
    real_time: bool
    velocity: int

    def __post_init__(self):
        # Sanity Check
        assert PIANO_A0 <= self.midi_number <= PIANO_C8, f"Note must be between A0 and C8, but found {self.midi_number}"
        assert self.duration >= 0, f"Duration must be greater than or equal to 0, but found {self.duration}"
        assert self.offset >= 0, f"Offset must be greater than or equal to 0, but found {self.offset}"
        assert 0 <= self.velocity < 128, f"Velocity must be between 0 and 127, but found {self.velocity}"

    def __repr__(self):
        return f"Note({self.note_name}, duration={self.duration}, offset={self.offset}, velocity={self.velocity})"

    @property
    def pitch_name(self) -> str:
        """Returns a note name of the pitch. e.g. A, C#, etc."""
        alter = self.alter
        if alter == 0:
            return self.step
        elif alter == 2:
            return f"{self.step}x"
        elif alter > 0:
            return f"{self.step}{'#' * alter}"
        else:
            return f"{self.step}{'b' * -alter}"

    @property
    def note_name(self):
        """The note name of the note. e.g. A4, C#5, etc."""
        return f"{self.pitch_name}{self.octave}"

    @property
    def step(self) -> Literal["C", "D", "E", "F", "G", "A", "B"]:
        """Returns the diatonic step of the note"""
        idx = self.index % 7
        return ("C", "G", "D", "A", "E", "B", "F")[idx]

    @property
    def step_number(self) -> int:
        """Returns the diatonic step number of the note, where C is 0, D is 1, etc."""
        idx = self.index % 7
        return (0, 4, 1, 5, 2, 6, 3)[idx]

    @property
    def alter(self):
        """Returns the alteration of the note aka number of sharps. Flats are represented as negative numbers."""
        return (self.index + 1) // 7

    @property
    def pitch_number(self):
        """Returns the chromatic pitch number of the note. C is 0, D is 2, etc. There are edge cases like B# returning 12 or Cb returning -1"""
        return ([0, 2, 4, 5, 7, 9, 11][self.step_number] + self.alter)

    @property
    def midi_number(self):
        """The chromatic pitch number of the note, using the convention that A4=440Hz converts to 69
        This is also the MIDI number of the note."""
        return self.pitch_number + 12 * self.octave + 12

    def transpose(self, interval: int, compound: int = 0) -> Note:
        """Transposes the note by a given interval. The interval is given by the relative LOF index.
        So unison is 0, perfect fifths is 1, major 3rds is 4, etc.
        Assuming transposing up. If you want to transpose down, say a perfect fifth,
        then transpose up a perfect fourth and compound by -1."""
        new_index = self.index + interval
        # Make a draft note to detect octave changes
        draft_note = Note(
            index=new_index,
            octave=self.octave,
            duration=self.duration,
            offset=self.offset,
            real_time=self.real_time,
            velocity=self.velocity
        )
        new_octave = self.octave + compound
        if (draft_note.pitch_number % 12) < (self.pitch_number % 12):
            new_octave += 1
        return Note(
            index=new_index,
            octave=new_octave,
            duration=self.duration,
            offset=self.offset,
            real_time=self.real_time,
            velocity=self.velocity
        )

    @classmethod
    def from_str(cls, note: str, real_time: bool = True) -> Note:
        """Creates a Note from a string note.

        Example: A4[0, 1, 64] is A in the 4th octave with a duration of 0 and offset of 1 and velocity of 64.
        A4[0, 1] is A in the 4th octave with a duration of 0 and offset of 1.
        A4 is A in the 4th octave with a duration of 0 and offset of 0.
        A is A in the (implied) 4th octave with a duration of 0 and offset of 0."""
        duration = 0
        offset = 0
        velocity = 64
        if "[" in note:
            note, rest = note.split("[")
            rest = rest.rstrip("]")
            assert len(rest.split(",")) in (2, 3), f"Rest must have 2 or 3 elements, but found {len(rest.split(','))}"
            if len(rest.split(",")) == 3:
                duration, offset, velocity = rest.split(",")
                duration = float(duration)
                offset = float(offset)
                velocity = int(velocity)
            else:
                duration, offset = rest.split(",")
                duration = float(duration)
                offset = float(offset)

        match = _PITCH_NAME_REGEX.match(note)
        if not match:
            # Add the implied octave
            match = _PITCH_NAME_REGEX.match(note + "4")

        assert match and len(match.groups()) == 3, f"The name {note} is not a valid note name"
        pitch_name, alter, octave = match.groups()
        if alter is None:
            alter = ""
        alter = alter.replace("x", "##").replace("-", "b").replace("+", "#")
        sharps = reduce(lambda x, y: x + 1 if y == "#" else x - 1, alter, 0)
        assert pitch_name in ("C", "D", "E", "F", "G", "A", "B"), f"Step must be one of CDEFGAB, but found {pitch_name}"  # to pass the typechecker

        return cls(
            index=_step_alter_to_lof_index(pitch_name, sharps),
            octave=int(octave),
            duration=duration,
            offset=offset,
            real_time=real_time,
            velocity=velocity
        )

    @classmethod
    def from_note(cls, note: m21.note.Note, real_time: bool = False, velocity: int = 64) -> Note:
        """Creates a Note from a music21 note."""
        alter = note.pitch.alter
        assert alter == int(alter), f"Alter must be an integer, but found {alter}"
        step = note.pitch.step
        octave = note.octave
        assert octave is not None, "Note must have an octave"
        duration = note.duration.quarterLength
        # If this note does not come from an active site, then the offset is (conveniently) 0
        offset = note.offset
        return cls(
            index=_step_alter_to_lof_index(step, int(alter)),
            octave=octave,
            duration=float(duration),
            offset=float(offset),
            real_time=real_time,
            velocity=velocity
        )

    @classmethod
    def from_midi_number(cls, midi_number: int, duration: float = 0., offset: float = 0., real_time: bool = True, velocity: int = 64) -> Note:
        """Creates a Note from a MIDI number. A4 maps to 69. If accidentals are needed, assumes the note is sharp."""
        octave = (midi_number // 12) - 1
        pitch = [0, 7, 2, 9, 4, -1, 6, 1, 8, 3, 10, 5][midi_number % 12]
        return cls(
            index=pitch,
            octave=octave,
            duration=duration,
            offset=offset,
            real_time=real_time,
            velocity=velocity
        )


class _Notes(SymbolicMusic):
    def __init__(self, notes: list[Note], real_time: bool):
        assert all(note.real_time == real_time for note in notes), f"All notes must be {'real' if real_time else 'notated'} time"
        self._notes = sorted(notes, key=lambda x: (x.offset, x.duration, x.midi_number))

    def __getitem__(self, index: int) -> Note:
        return self._notes[index]

    def __iter__(self):
        return iter(self._notes)

    def __bool__(self):
        return len(self._notes) > 0
    
    def __len__(self):
        return len(self._notes)

    def normalize(self):
        min_offset = min(note.offset for note in self._notes)
        for note in self._notes:
            # Use python black magic - this is safe because the object only has reference here
            object.__setattr__(note, "offset", note.offset - min_offset)
        return self

    def to_pianoroll(self, resolution: int = 24, eps: float = 1e-6) -> PianoRoll:
        """Converts the notes to a pianoroll. A real-timed list of notes will be converted to a real-timed pianoroll and vice versa."""
        raise NotImplementedError

    def save_to_midi(self, path: str):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        ticks_per_beat = mid.ticks_per_beat  # Default 480
        tempo = mido.bpm2tempo(120)  # Converts BPM to microseconds per beat, default is 120 BPM

        events = []
        for note in self:
            on = mido.second2tick(note.offset, ticks_per_beat, tempo)
            off = mido.second2tick(note.offset + note.duration, ticks_per_beat, tempo)
            events.append((on, note.midi_number, 'note_on', note.velocity))
            events.append((off, note.midi_number, 'note_off', 0))

        events.sort()
        current_time = 0
        for event in events:
            time, note, tp, velocity = event
            delta_ticks = time - current_time
            track.append(Message(tp, note=note, velocity=velocity, time=delta_ticks))
            current_time = time

        mid.save(path)


class RealTimeNotes(_Notes):
    def __init__(self, notes: list[Note]):
        super().__init__(notes, real_time=True)

    def __add__(self, other: RealTimeNotes) -> RealTimeNotes:
        return RealTimeNotes(self._notes + other._notes)

    def to_pianoroll(self, resolution: int = 24, eps: float = 1e-6):
        """Converts the notes to a real-time pianoroll."""
        return notes_to_pianoroll_rt(self, resolution, eps)

    @classmethod
    def load_from_xml(cls, path: str) -> RealTimeNotes:
        """Loads a RealTimeNotes from a music21 XML file."""
        from .data import Midifile
        return Midifile.load_from_xml(path).to_real_time_notes()


class NotatedTimeNotes(_Notes):
    def __init__(self, notes: list[Note]):
        super().__init__(notes, real_time=False)

    def __add__(self, other: NotatedTimeNotes) -> NotatedTimeNotes:
        return NotatedTimeNotes(self._notes + other._notes)

    def to_pianoroll(self, resolution: int = 24, eps: float = 1e-6):
        """Converts the notes to a notated-time pianoroll."""
        return notes_to_pianoroll_nt(self, resolution, eps)

    @classmethod
    def load_from_xml(cls, path: str) -> NotatedTimeNotes:
        """Loads a NotatedTimeNotes from a music21 XML file."""
        from .data import Music21Stream
        return Music21Stream.load_from_xml(path).to_notated_time_notes()

    def to_real_time(self, tempo: float = 120.) -> RealTimeNotes:
        notes = self._notes
        assert all(not note.real_time for note in notes), "All notes must be timed against quarter length"
        return RealTimeNotes([Note(
            index=n.index,
            octave=n.octave,
            duration=n.duration * 60 / tempo,
            offset=n.offset * 60 / tempo,
            real_time=True,
            velocity=n.velocity
        ) for n in notes])


def _step_alter_to_lof_index(step: Literal["C", "D", "E", "F", "G", "A", "B"], alter: int) -> int:
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter


def notes_to_pianoroll_rt(notes: RealTimeNotes, resolution: int = 24, eps: float = 1e-6):
    """Converts a list of notes to a pianoroll. A real-timed list of notes will be converted to a real-timed pianoroll and vice versa."""
    from .roll import RealTimePianoRoll

    if not notes:
        raise ValueError("Cannot convert an empty list of notes to a pianoroll")
    assert all(note.real_time == notes[0].real_time for note in notes), "All notes must have the same timing property"

    max_duration = max(note.offset + note.duration for note in notes)
    max_duration = int(max_duration * resolution) + 1
    pianoroll = RealTimePianoRoll.new_zero_array(max_duration)
    notes_dict: dict[int, list[tuple[int, int, int]]] = {}
    for note in notes:
        start = int(note.offset * resolution)
        end = int((note.offset + note.duration) * resolution)
        if start == end and note.duration > 0:
            required_resolution = int(1 / note.duration)
            raise ValueError(f"Unable to resolve piano roll - try increase the resolution to at least {required_resolution}")
        if note.midi_number not in notes_dict:
            notes_dict[note.midi_number] = []
        notes_dict[note.midi_number].append((start, end, note.velocity))

    # Check for overlap first, then assign the notes
    for k, ns in notes_dict.items():
        ns = sorted(ns, key=lambda x: x[0])
        for i in range(len(ns) - 1):
            if ns[i][1] < ns[i + 1][0]:
                # Case 1: No overlap
                pianoroll[ns[i][0]:ns[i][1], k - PIANO_A0] = ns[i][2] / 127
            elif abs(ns[i][1] - ns[i + 1][0]) < eps:
                # Case 2: End of note is start of next note - try minus one on end of this note
                if ns[i][1] == ns[i][0] + 1:
                    raise ValueError("Unable to resolve piano roll - try increase the resolution")
                pianoroll[ns[i][0]:ns[i][1] - 1, k - PIANO_A0] = ns[i][2] / 127
            else:
                # Case 3: Overlap
                # Use the union of two intervals and the larger velocity
                pianoroll[min(ns[i][0], ns[i + 1][0]):max(ns[i][1], ns[i + 1][1]), k - PIANO_A0] = max(ns[i][2], ns[i + 1][2]) / 127
        pianoroll[ns[-1][0]:ns[-1][1], k - PIANO_A0] = ns[-1][2] / 127

    return RealTimePianoRoll(pianoroll, resolution, notes[0].real_time)


def notes_to_pianoroll_nt(notes: NotatedTimeNotes, resolution: int = 24, eps: float = 1e-6):
    """Converts a list of notes to a pianoroll. A real-timed list of notes will be converted to a real-timed pianoroll and vice versa."""
    from .roll import NotatedPianoRoll

    if not notes:
        raise ValueError("Cannot convert an empty list of notes to a pianoroll")
    assert all(note.real_time == notes[0].real_time for note in notes), "All notes must have the same timing property"

    max_duration = max(note.offset + note.duration for note in notes)
    max_duration = int(max_duration * resolution) + 1
    pianoroll = NotatedPianoRoll.new_zero_array(max_duration)
    notes_dict: dict[int, list[tuple[int, int, int]]] = {}
    for note in notes:
        start = int(note.offset * resolution)
        end = int((note.offset + note.duration) * resolution)
        if start == end and note.duration > 0:
            required_resolution = int(1 / note.duration)
            raise ValueError(f"Unable to resolve piano roll - try increase the resolution to at least {required_resolution}")
        if note.midi_number not in notes_dict:
            notes_dict[note.midi_number] = []
        notes_dict[note.midi_number].append((start, end, note.velocity))

    # Check for overlap first, then assign the notes
    for k, ns in notes_dict.items():
        ns = sorted(ns, key=lambda x: x[0])
        for i in range(len(ns) - 1):
            if ns[i][1] < ns[i + 1][0]:
                # Case 1: No overlap
                pianoroll[ns[i][0]:ns[i][1], k - PIANO_A0] = ns[i][2] / 127
            elif abs(ns[i][1] - ns[i + 1][0]) < eps:
                # Case 2: End of note is start of next note - try minus one on end of this note
                if ns[i][1] == ns[i][0] + 1:
                    raise ValueError("Unable to resolve piano roll - try increase the resolution")
                pianoroll[ns[i][0]:ns[i][1] - 1, k - PIANO_A0] = ns[i][2] / 127
            else:
                # Case 3: Overlap
                # Use the union of two intervals and the larger velocity
                pianoroll[min(ns[i][0], ns[i + 1][0]):max(ns[i][1], ns[i + 1][1]), k - PIANO_A0] = max(ns[i][2], ns[i + 1][2]) / 127
        pianoroll[ns[-1][0]:ns[-1][1], k - PIANO_A0] = ns[-1][2] / 127

    return NotatedPianoRoll(pianoroll, resolution, notes[0].real_time)
