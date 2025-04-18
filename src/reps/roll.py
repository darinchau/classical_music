import numpy as np
import typing
from numpy.typing import NDArray
import warnings
import librosa
from ..util import PIANO_A0, PIANO_C8
from ..util import is_ipython
from .base import SymbolicMusic
from .notes import Note, RealTimeNotes, NotatedTimeNotes

import librosa.display


class _PianoRoll(SymbolicMusic):
    """A piano roll is defined as a 2D matrix (T, 90) where T is the number of time steps and 88 piano keys + 2 pedals is the feature vectors.
    The roll r[i, j] represents the strength of the jth piano key being pressed at time i.
    By convention, 0-88 is the piano keys, r[:, 88] is the sustain pedal, and r[:, 89] is the soft pedal.
    """

    def __init__(self, pianoroll: NDArray[np.float32], resolution: int = 24, real_time: bool = True):
        assert np.all(pianoroll >= 0) and np.all(pianoroll <= 1), "Pianoroll must be between 0 and 1"
        assert pianoroll.shape[1] == 90, "Pianoroll must have 90 features"
        assert resolution > 0, "Resolution must be greater than 0"
        self._pianoroll = pianoroll
        self._resolution = resolution
        self._pianoroll.flags.writeable = False

    @property
    def resolution(self) -> int:
        return self._resolution

    @property
    def pianoroll(self) -> NDArray:
        return self._pianoroll

    @property
    def implied_duration(self) -> float:
        """The duration of the pianoroll as suggested by the shape of the pianoroll"""
        return self._pianoroll.shape[0] / self._resolution

    @property
    def real_time(self) -> bool:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._pianoroll.shape)  # type: ignore

    @staticmethod
    def new_zero_array(nframes: int) -> NDArray:
        return np.zeros((nframes, 90), dtype=np.float32)

    def plot(self, **kwargs):
        """Plots the pianoroll"""
        # Hijack the specshow function to plot the pianoroll
        # Since chroma cqt plots are just fancy pianorolls
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            librosa.display.specshow(
                self._pianoroll.T,
                sr=self._resolution,
                x_axis='time',
                y_axis='cqt_note',
                fmin=librosa.midi_to_hz(21).item(),
                hop_length=1,
            )


class RealTimePianoRoll(_PianoRoll):
    """A real time piano roll is a piano roll that has a fixed resolution and is not time-stretched."""
    @property
    def real_time(self) -> bool:
        return True

    def to_real_time_notes(self) -> RealTimeNotes:
        """Converts the pianoroll to a list of notes"""
        note = pianoroll_to_notes(self)
        assert isinstance(note, RealTimeNotes), "Pianoroll is not in real time"
        return note

    @classmethod
    def load_from_xml(cls, path: str):
        """Loads a pianoroll from an XML file"""
        return RealTimeNotes.load_from_xml(path).to_pianoroll()

    def save_to_midi(self, path: str):
        """Saves the pianoroll to a MIDI file"""
        return self.to_real_time_notes().save_to_midi(path)


class NotatedPianoRoll(_PianoRoll):
    """A notated piano roll is a piano roll that has a fixed resolution and is time-stretched."""
    @property
    def real_time(self) -> bool:
        return False

    def to_notated_notes(self) -> NotatedTimeNotes:
        """Converts the pianoroll to a list of notes"""
        note = pianoroll_to_notes(self)
        assert isinstance(note, NotatedTimeNotes), "Pianoroll is not in real time"
        return note

    @classmethod
    def load_from_xml(cls, path: str):
        """Loads a pianoroll from an XML file"""
        return NotatedTimeNotes.load_from_xml(path).to_pianoroll()

    def save_to_midi(self, path: str):
        """Saves the pianoroll to a MIDI file"""
        return self.to_notated_notes().save_to_midi(path)


def _check_pianoroll_fail_reason(array: NDArray, raise_error: bool = False) -> typing.Optional[str]:
    def _error(error_message: str):
        if raise_error:
            raise ValueError(error_message)
        return error_message

    if not array.shape == 2:
        return _error("Pianoroll must have 2 dimensions")

    if not array.shape[1] == 90:
        return _error("Pianoroll must have 90 features")

    if not np.all(array >= 0) or not np.all(array <= 1):
        return _error("Pianoroll must be between 0 and 1")

    # Note - Velocity pairs
    note_on_dict = {}
    total_time = array.shape[0]
    for i in range(total_time):
        for note in range(88):
            # Pedals are ignored - only process notes
            if array[i, note] == 0 and note in note_on_dict:
                note_on_dict.pop(note)
            elif array[i, note] > 0 and note in note_on_dict:
                if not np.isclose(array[i, note], note_on_dict[note]):
                    return _error(f"Note {note} (t = {i}) has a velocity of {array[i, note]} but expected {note_on_dict[note]}")
            elif array[i, note] > 0 and note not in note_on_dict:
                note_on_dict[note] = array[i, note]
    return None


def pianoroll_to_notes(pianoroll: _PianoRoll) -> RealTimeNotes | NotatedTimeNotes:
    """Converts a pianoroll to a list of notes. A list of notes with the same timing property will be returned."""
    _check_pianoroll_fail_reason(pianoroll.pianoroll, raise_error=True)
    array = pianoroll.pianoroll
    note_on_dict = {}
    total_time = array.shape[0]
    notes: list[Note] = []
    for i in range(total_time):
        for note in range(90):
            # Pedals are ignored - only process notes
            if array[i, note] == 0 and note in note_on_dict:
                start_time, velocity = note_on_dict.pop(note)
                duration = i - start_time
                note = Note.from_midi_number(
                    midi_number=note + PIANO_A0,
                    duration=duration / pianoroll.resolution,
                    offset=start_time / pianoroll.resolution,
                    real_time=pianoroll.real_time,
                    velocity=int(velocity * 127)
                )
                notes.append(note)
            elif array[i, note] > 0 and note not in note_on_dict:
                note_on_dict[note] = (i, array[i, note])
    if pianoroll.real_time:
        return RealTimeNotes(notes)
    return NotatedTimeNotes(notes)
