from __future__ import annotations
import os
import copy
import heapq
import librosa
import logging
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mido
import music21 as m21
import numpy as np
import random
import re
import soundfile as sf
import subprocess
import tempfile
import threading
import typing
import warnings
from .util import NATURAL, is_ipython
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import lru_cache, reduce, total_ordering
from math import pi as PI
from music21.midi.translate import streamToMidiFile
from music21 import expressions, style, stream
from music21.stream.base import Opus
from music21.ipython21 import converters as ip21_converters
from music21.converter.subConverters import ConverterMusicXML
from music21 import defaults
from music21.converter import museScore
from mido import MidiFile, MidiTrack, Message
from numpy.typing import NDArray
from typing import Literal

if typing.TYPE_CHECKING:
    from .player import NotesPlayer


_PITCH_NAME_REGEX = re.compile(r"([CDEFGAB])(#+|b+)?(-?[0-9]+)")
PIANO_A0 = 21
PIANO_C8 = 108

_music21_setup = False


class NotSupportedOnWindows(NotImplementedError):
    pass


def _get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("You need to install sounddevice to use the play function")


def _setup():
    from music21 import environment
    global _music21_setup
    if _music21_setup:
        return

    # Raise a warning if in windows
    if os.name == "nt":
        raise NotSupportedOnWindows("Music21 is not fully supported in Windows. Please use Linux or MacOS for better compatibility")

    us = environment.UserSettings()
    us['musescoreDirectPNGPath'] = '/usr/bin/mscore'
    us['directoryScratch'] = '/tmp'

    _music21_setup = True


class MusicRepresentation(ABC):
    """Base class to all music representations. Currently does nothing."""


class Audio(MusicRepresentation):
    """Represents an audio that can be played and saved.

    A stripped down version of Audio class from https://github.com/darinchau/AutoMasher
    that uses numpy array instead of pytorch tensor so we don't have to install pytorch"""

    def sanity_check(self):
        assert self._sample_rate > 0
        assert isinstance(self._sample_rate, int)
        assert len(self._data.shape) == 2
        assert 1 <= self._data.shape[0] <= 2
        assert isinstance(self._data, np.ndarray)
        assert self._data.dtype == self.dtype()
        assert self._inited
        self._data.flags.writeable = False

    def __init__(self, data: NDArray[np.float32], sample_rate: int):
        """An audio is a special type of audio features - each feature vector has 1 dimensions"""
        assert len(data.shape) == 2, f"Audio data must have 2 dimensions, but found {data.shape}"
        assert data.dtype == self.dtype(), f"Audio data must have dtype {self.dtype()} but found {data.dtype}"
        assert sample_rate > 0, "Sample rate must be greater than 0"

        self._data = data
        self._sample_rate = sample_rate
        self._data.flags.writeable = False
        self._inited = True
        self.sanity_check()

        # For playing audio
        self._stop_audio = False
        self._thread = None

    @staticmethod
    def dtype():
        return np.float32

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate of the audio"""
        self.sanity_check()
        return self._sample_rate

    @property
    def nframes(self) -> int:
        """Returns the number of frames of the audio"""
        self.sanity_check()
        return self._data.shape[1]

    @property
    def nchannels(self) -> int:
        """Returns the number of channels of the audio"""
        self.sanity_check()
        return self._data.shape[0]

    @property
    def duration(self) -> float:
        """Returns the duration of the audio in seconds"""
        self.sanity_check()
        return self.nframes / self.sample_rate

    @property
    def data(self):
        """Returns the audio data array"""
        self.sanity_check()
        return self._data

    @classmethod
    def load(cls, fpath: str) -> Audio:
        """
        Loads an audio file from a given file path, and returns the audio as a tensor.
        Output shape: (channels, N) where N = duration (seconds) x sample rate (hz)

        if channels == 1, then take the mean across the audio
        if channels == audio channels, then leave it alone
        otherwise we will take the mean and duplicate the tensor until we get the desired number of channels

        Cache Path will be ignored if the file path is not a youtube url
        """
        wav, sr = librosa.load(fpath, mono=False)
        sr = int(sr)
        if len(wav.shape) > 1:
            wav = wav.reshape(-1, wav.shape[-1])
        else:
            wav = wav.reshape(1, -1)
        return cls(wav, sr)

    def play(self, blocking: bool = False, info: list[tuple[str, float]] | None = None):
        """Plays audio in a separate thread. Use the stop() function or wait() function to let the audio stop playing.
        info is a list of stuff you want to print. Each element is a tuple of (str, float) where the float is the time in seconds
        if progress is true, then display a nice little bar that shows the progress of the audio"""
        sd = _get_sounddevice()

        def _play(sound, sr, nc, stop_event):
            event = threading.Event()
            x = 0

            def callback(outdata, frames, time, status):
                nonlocal x
                sound_ = sound[x:x+frames]
                x = x + frames

                # Print the info if there are anything
                while info and x/sr > info[0][1]:
                    info_str = info[0][0].ljust(longest_info)
                    print("\r" + info_str, end="")
                    info.pop(0)

                if stop_event():
                    raise sd.CallbackStop

                # Push the audio
                if len(outdata) > len(sound_):
                    outdata[:len(sound_)] = sound_
                    outdata[len(sound_):] = np.zeros((len(outdata) - len(sound_), 1))
                    raise sd.CallbackStop
                else:
                    outdata[:] = sound_[:]

            stream = sd.OutputStream(samplerate=sr, channels=nc, callback=callback, blocksize=1024, finished_callback=event.set)
            with stream:
                event.wait()
                self._stop_audio = True

        if info is not None:
            blocking = True  # Otherwise jupyter notebook will behave weirdly
        else:
            if is_ipython():
                from IPython.display import Audio as IPAudio  # type: ignore
                return IPAudio(self._data, rate=self.sample_rate)
            info = []
        info = sorted(info, key=lambda x: x[1])
        longest_info = max([len(x[0]) for x in info]) if info else 0
        sound = self._data.T
        self._thread = threading.Thread(target=_play, args=(sound, self.sample_rate, self.nchannels, lambda: self._stop_audio))
        self._stop_audio = False
        self._thread.start()
        if blocking:
            self.wait()

    def stop(self):
        """Attempts to stop the audio that's currently playing. If the audio is not playing, this does nothing."""
        self._stop_audio = True
        self.wait()

    def wait(self):
        """Wait for the audio to stop playing. If the audio is not playing, this does nothing."""
        if self._thread is None:
            return

        if not self._thread.is_alive():
            return

        self._thread.join()
        self._thread = None
        self._stop_audio = False  # Reset the state

    def save(self, fpath: str):
        """Saves the audio at the provided file path. WAV is (almost certainly) guaranteed to work"""
        self.sanity_check()
        data = self._data
        if fpath.endswith(".mp3"):
            try:
                from pydub import AudioSegment
            except ImportError:
                raise RuntimeError("You need to install pydub to save the audio as mp3")
            with tempfile.TemporaryDirectory() as tempdir:
                temp_fpath = os.path.join(tempdir, "temp.wav")
                sf.write(temp_fpath, data.T, self._sample_rate, format="wav")
                song = AudioSegment.from_wav(temp_fpath)
                song.export(fpath, format="mp3")
            return
        try:
            sf.write(fpath, data.T, self._sample_rate, format="wav")
            return
        except (ValueError, RuntimeError) as e:  # Seems like torchaudio changed the error type to runtime error in 2.2?
            # or the file path is invalid
            raise RuntimeError(f"Error saving the audio: {e} - {fpath}")

    def __repr__(self):
        """
        Prints out the following information about the audio:
        Duration, Sample rate, Num channels, Num frames
        """
        return f"(Audio)\nDuration:\t{self.duration:5f}\nSample Rate:\t{self.sample_rate}\nChannels:\t{self.nchannels}\nNum frames:\t{self.nframes}"

    def slice_frames(self, start_frame: int = 0, end_frame: int = -1) -> Audio:
        """Takes the current audio and splice the audio between start (frames) and end (frames). Returns a new copy.

        Specify end = -1 to take everything alll the way until the end"""
        assert start_frame >= 0
        assert end_frame == -1 or (end_frame > start_frame and end_frame <= self.nframes)
        data = None

        if end_frame == -1:
            data = self._data[:, start_frame:]
        if end_frame > 0:
            data = self._data[:, start_frame:end_frame]

        assert data is not None
        return Audio(data.copy(), self.sample_rate)

    def slice_seconds(self, start: float = 0, end: float = -1) -> Audio:
        """Takes the current audio and splice the audio between start (seconds) and end (seconds). Returns a new copy.

        Specify end = -1 to take everything alll the way until the end"""
        assert start >= 0
        start_frame = int(start * self._sample_rate)
        end_frame = self.nframes if end == -1 else int(end * self._sample_rate)
        assert start_frame < end_frame <= self.nframes
        if end_frame == self.nframes:
            end_frame = -1
        return self.slice_frames(start_frame, end_frame)

    def resample(self, sample_rate: int) -> Audio:
        """Resamples the audio to a new sample rate"""
        assert sample_rate > 0
        data = librosa.resample(self._data, orig_sr=self.sample_rate, target_sr=sample_rate)
        return Audio(data, sample_rate)

    def pad(self, target: int, front: bool = False) -> Audio:
        """Returns a new audio with the given number of frames and the same sample rate as self.
        If n < self.nframes, we will trim the audio; if n > self.nframes, we will perform zero padding
        If front is set to true, then operate on the front instead of on the back"""
        length = self.nframes
        if not front:
            if length > target:
                new_data = self._data[:, :target].copy()
            else:
                new_data = np.pad(self._data, ((0, 0), (0, target - length)), mode='constant')
        else:
            if length > target:
                new_data = self._data[:, -target:].copy()
            else:
                new_data = np.pad(self._data, ((0, 0), (target - length, 0)), mode='constant')

        assert new_data.shape[1] == target

        return Audio(new_data, self._sample_rate)

    def to_nchannels(self, target: typing.Literal[1, 2]) -> Audio:
        """Return self with the correct target. If you use int, you must guarantee the value is 1 or 2, otherwise you get an error"""
        self.sanity_check()
        if self.nchannels == target:
            return Audio(self.data.copy(), self.sample_rate)

        if self.nchannels == 1 and target == 2:
            return Audio(np.stack([self.data[0], self.data[0]], axis=0), self.sample_rate)

        if self.nchannels == 2 and target == 1:
            return Audio(self.data.mean(axis=0, keepdims=True), self.sample_rate)

        assert False, "Unreachable"

    def plot(self):
        waveform = self.data
        num_channels = self.nchannels
        num_frames = self.nframes

        time_axis = np.arange(0, num_frames) / self.sample_rate

        figure, axes = plt.subplots()
        if num_channels == 1:
            axes.plot(time_axis, waveform[0], linewidth=1)
        else:
            axes.plot(time_axis, np.abs(waveform[0]), linewidth=1)
            axes.plot(time_axis, -np.abs(waveform[1]), linewidth=1)
        axes.grid(True)
        plt.show(block=False)


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


class _Notes(MusicRepresentation):
    def __init__(self, notes: list[Note], real_time: bool):
        assert all(note.real_time == real_time for note in notes), f"All notes must be {'real' if real_time else 'notated'} time"
        self._notes = copy.deepcopy(notes)

    def __getitem__(self, index: int) -> Note:
        return self._notes[index]

    def __iter__(self):
        return iter(self._notes)

    def __bool__(self):
        return len(self._notes) > 0


class RealTimeNotes(_Notes):
    def __init__(self, notes: list[Note]):
        super().__init__(notes, real_time=True)

    def __add__(self, other: RealTimeNotes) -> RealTimeNotes:
        return RealTimeNotes(self._notes + other._notes)


class NotatedTimeNotes(_Notes):
    def __init__(self, notes: list[Note]):
        super().__init__(notes, real_time=False)

    def __add__(self, other: NotatedTimeNotes) -> NotatedTimeNotes:
        return NotatedTimeNotes(self._notes + other._notes)


class PianoRoll(MusicRepresentation):
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
        self._real_time = real_time
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
        return self._real_time

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





def _step_alter_to_lof_index(step: Literal["C", "D", "E", "F", "G", "A", "B"], alter: int) -> int:
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter


def _convert_midi_to_wav(input_path: str, output_path: str, soundfont_path="~/.fluidsynth/default_sound_font.sf2", sample_rate=44100, verbose=False):
    assert _is_package_installed("fluidsynth"), "You need to install fluidsynth to convert midi to audio, refer to README for more details"
    subprocess.call(['fluidsynth', '-ni', soundfont_path, input_path, '-F', output_path, '-r', str(sample_rate)],
                    stdout=subprocess.DEVNULL if not verbose else None,
                    stderr=subprocess.DEVNULL if not verbose else None)


def _is_package_installed(package_name):
    if os.name == "nt":
        raise NotSupportedOnWindows(f"The package ``{package_name}`` is not supported in Windows")
    try:
        result = subprocess.run(['dpkg', '-s', package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if 'install ok installed' in result.stdout:
            return True
    except subprocess.CalledProcessError:
        return False
    return False


def score_to_notes(stream: m21.stream.Score) -> NotatedTimeNotes:
    notes: list[Note] = []
    for el in stream.recurse().getElementsByClass((
        m21.note.Note,
        m21.chord.Chord,
    )):
        # Calculate the offset
        x: m21.base.Music21Object = el
        offset = Fraction()
        while x.activeSite is not None:
            offset += x.offset
            x = x.activeSite
            if x.id == stream.id:
                break
        else:
            assert False, f"Element {el} is not in the score"
        offset = float(offset)

        # Append the note or chord
        if isinstance(el, m21.note.Note):
            note = Note.from_note(el, real_time=False)
            # Use some python black magic to ensure the offset is calculated correctly
            object.__setattr__(note, "offset", offset)
            notes.append(note)
        elif isinstance(el, m21.chord.Chord):
            for el_ in el.notes:
                note = Note.from_note(el_, real_time=False)
                # Use some python black magic to ensure the offset is calculated correctly
                object.__setattr__(note, "offset", offset)
                notes.append(note)
    return NotatedTimeNotes(notes)


def _midi_to_notes_quarter_length(midi_path: str) -> NotatedTimeNotes:
    # Use music21 to convert the midi to notes
    if not _music21_setup:
        _setup()
    stream = m21.converter.parse(midi_path)
    if not isinstance(stream, m21.stream.Score):
        raise ValueError(f"Midi file must contain a score, found {type(stream)}")
    return score_to_notes(stream)


def _midi_to_notes_real_time(midi_path: str) -> RealTimeNotes:
    mid = mido.MidiFile(midi_path)
    tempo = 500000  # Default tempo (500,000 microseconds per beat)
    ticks_per_beat = mid.ticks_per_beat

    tempo_changes = [(0, tempo)]
    events = []

    # Convert delta times to absolute times and collect all events
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_changes.append((current_tick, msg.tempo))
            if msg.type in ['note_on', 'note_off']:
                events.append((current_tick, msg.note, msg.type, msg.velocity))

    tempo_changes.sort()
    events.sort()
    notes: list[Note] = []
    current_time = 0
    last_tick = 0
    note_on_dict: dict[int, tuple[float, int]] = {}

    # Convert ticks in events to real time using the global tempo map
    for event in events:
        tick, note, tp, velocity = event

        while tempo_changes and tempo_changes[0][0] <= tick:
            tempo_change_tick, new_tempo = tempo_changes.pop(0)
            if tempo_change_tick > last_tick:
                current_time += mido.tick2second(tempo_change_tick - last_tick, ticks_per_beat, tempo)
                last_tick = tempo_change_tick
            tempo = new_tempo

        # Update current time up to the event tick
        if tick > last_tick:
            current_time += mido.tick2second(tick - last_tick, ticks_per_beat, tempo)
            last_tick = tick

        if tp == 'note_on' and velocity > 0:
            note_on_dict[note] = (current_time, velocity)
        elif (tp == 'note_off' or (tp == 'note_on' and velocity == 0)) and note in note_on_dict:
            start_time, velocity = note_on_dict.pop(note)
            duration = current_time - start_time
            note = Note.from_midi_number(midi_number=note, duration=duration, offset=start_time, velocity=velocity, real_time=True)
            notes.append(note)

    return RealTimeNotes(notes)


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


def notes_to_real_time(notes: NotatedTimeNotes, tempo: float = 120.) -> RealTimeNotes:
    """Converts notes to real time"""
    assert all(not note.real_time for note in notes), "All notes must be timed against quarter length"
    return RealTimeNotes([Note(
        index=n.index,
        octave=n.octave,
        duration=n.duration * 60 / tempo,
        offset=n.offset * 60 / tempo,
        real_time=True,
        velocity=n.velocity
    ) for n in notes])


def display_score(obj: m21.base.Music21Object, invert_color: bool = True, skip_display: bool = False):
    """Displays the score. Returns a dictionary where keys are the page numbers and values are the images of the page in np arrays"""
    if not _music21_setup:
        _setup()

    from music21.stream.base import Opus
    from music21.converter.subConverters import ConverterMusicXML
    from music21 import defaults
    from music21.converter import museScore

    savedDefaultTitle = defaults.title
    savedDefaultAuthor = defaults.author
    defaults.title = ''
    defaults.author = ''

    if isinstance(obj, Opus):
        raise NotImplementedError("Perform a recursive call to show here when we support Opuses. Ref: music21.ipython21.converters.showImageThroughMuseScore")

    fp = ConverterMusicXML().write(
        obj,
        fmt="musicxml",
        subformats=["png"],
        trimEdges=True,
    )

    last_png = museScore.findLastPNGPath(fp)
    last_number, num_digits = museScore.findPNGRange(fp, last_png)

    pages: dict[int, np.ndarray] = {}
    stem = str(fp)[:str(fp).rfind('-')]
    for pg in range(1, last_number + 1):
        page_str = stem + '-' + str(pg).zfill(num_digits) + '.png'
        page = np.array(mpimg.imread(page_str) * 255, dtype=np.uint8)

        # Invert the color because dark mode
        if invert_color:
            page[:, :, :3] = 255 - page[:, :, :3]
        pages[pg] = page

    if is_ipython() and not skip_display:
        from IPython.display import Image, display, HTML  # type: ignore

        for pg in range(1, last_number + 1):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                mpimg.imsave(f.name, pages[pg])
                display(Image(data=f.read(), retina=True))
            if pg < last_number:
                display(HTML('<p style="padding-top: 20px">&nbsp;</p>'))

    defaults.title = savedDefaultTitle
    defaults.author = savedDefaultAuthor
    return pages


def _midi_to_notes(midi_path: str, real_time: bool = True, normalize: bool = False):
    """Converts a midi file to a list of notes. If real_time is True, then the notes will be timed against real time in seconds.

    If normalize is True, then the earliest note will always have an offset of 0."""
    if real_time:
        notes = _midi_to_notes_real_time(midi_path)
    else:
        notes = _midi_to_notes_quarter_length(midi_path)
    assert all(note.real_time == real_time for note in notes)
    notes._notes = sorted(notes._notes, key=lambda x: x.offset)
    if normalize:
        min_offset = min(note.offset for note in notes._notes)
        for note in notes._notes:
            # Use python black magic - this is safe because the object only has reference here
            object.__setattr__(note, "offset", note.offset - min_offset)
    return notes


def midi_to_real_time_notes(midi_path: str) -> RealTimeNotes:
    notes = _midi_to_notes(midi_path, real_time=True)
    assert isinstance(notes, RealTimeNotes)
    return notes


def midi_to_notated_time_notes(midi_path: str) -> NotatedTimeNotes:
    notes = _midi_to_notes(midi_path, real_time=False)
    assert isinstance(notes, NotatedTimeNotes)
    return notes


def score_to_audio(score: m21.stream.Score, sample_rate: int = 44100, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2") -> Audio:
    """Inner helper function to convert a music21 score to audio. The score will be consumed."""
    if not _music21_setup:
        _setup()
    with (
        tempfile.NamedTemporaryFile(suffix=".mid") as f1,
        tempfile.NamedTemporaryFile(suffix=".wav") as f2
    ):
        file = streamToMidiFile(score, addStartDelay=True)
        file.open(f1.name, "wb")
        try:
            file.write()
        finally:
            file.close()
        _convert_midi_to_wav(f1.name, f2.name, soundfont_path, sample_rate)
        return Audio.load(f2.name)


def midi_to_audio(midi_path: str, sample_rate: int = 44100, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2") -> Audio:
    """Converts MIDI to audio. This function retains instrument information. To make everything into piano, use midi_to_notes and then notes_to_audio"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        _convert_midi_to_wav(midi_path, f.name, soundfont_path, sample_rate)
        return Audio.load(f.name)


def notes_to_score(notes: NotatedTimeNotes) -> m21.stream.Score:
    """Convert a list of notes to a music21 score. The score is only intended to be played and not for further analysis."""
    if not _music21_setup:
        _setup()

    score = m21.stream.Score()

    listnotes = sorted(notes._notes, key=lambda x: x.offset)
    heap = []

    # This dictionary will store the clef assignments (clef number -> list of events)
    assignments: dict[int, list[Note]] = {}
    clef_ctr = 0

    # Iterate through the sorted list of events
    for note in listnotes:
        end = note.offset + note.duration
        if heap and heap[0][0] <= note.offset:
            _, clef_to_use = heapq.heappop(heap)
        else:
            clef_to_use = clef_ctr
            clef_ctr += 1

        if clef_to_use in assignments:
            assignments[clef_to_use].append(note)
        else:
            assignments[clef_to_use] = [note]
        heapq.heappush(heap, (end, clef_to_use))

    measures = [m21.stream.Measure() for _ in range(len(assignments))]
    parts = [m21.stream.Part() for _ in range(len(assignments))]
    for i, (measure, part) in enumerate(zip(measures, parts)):
        for note in assignments[i]:
            m21note = m21.note.Note(
                note.note_name,
                quarterLength=note.duration
            )
            m21note.volume.velocity = note.velocity
            measure.append(m21note)
        part.append(measure)
        score.append(part)
        part.makeRests(inPlace=True, fillGaps=True)
    return score


def notes_to_pianoroll(notes: RealTimeNotes | NotatedTimeNotes, resolution: int = 24, eps: float = 1e-6) -> PianoRoll:
    """Converts a list of notes to a pianoroll. A real-timed list of notes will be converted to a real-timed pianoroll and vice versa."""
    if not notes:
        raise ValueError("Cannot convert an empty list of notes to a pianoroll")
    assert all(note.real_time == notes[0].real_time for note in notes), "All notes must have the same timing property"

    max_duration = max(note.offset + note.duration for note in notes)
    max_duration = int(max_duration * resolution) + 1
    pianoroll = PianoRoll.new_zero_array(max_duration)
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

    return PianoRoll(pianoroll, resolution, notes[0].real_time)


def notes_to_midi(notes: NotatedTimeNotes | RealTimeNotes, fpath: str):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_beat = mid.ticks_per_beat  # Default 480
    tempo = mido.bpm2tempo(120)  # Converts BPM to microseconds per beat, default is 120 BPM

    events = []
    for note in notes:
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

    mid.save(fpath)


def pianoroll_to_notes(pianoroll: PianoRoll) -> RealTimeNotes | NotatedTimeNotes:
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


def notes_to_audio(notes: RealTimeNotes | NotatedTimeNotes, player: NotesPlayer | None = None, tempo: float = 120., sample_rate: int = 48000) -> Audio:
    """Turns the Notes into Audio. If the notes are timed in real time then the tempo will be ignored.
    Otherwise the notes will be converted to real time using the tempo provided.

    By default, this function uses the fluid synth player. If you want to use a different player, then provide the player object."""
    assert notes and all(note.real_time == notes[0].real_time for note in notes), "All notes must have the same timing property"
    from .player import FluidSynthNotesPlayer, NotesPlayer
    if isinstance(notes, NotatedTimeNotes):
        notes = notes_to_real_time(notes, tempo)
    if player is None:
        player = FluidSynthNotesPlayer()
    audio = player.play(notes, sample_rate=sample_rate)
    if audio.sample_rate != sample_rate:
        audio = audio.resample(sample_rate)
    return audio
