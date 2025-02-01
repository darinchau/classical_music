from __future__ import annotations
import os
import heapq
import librosa
import logging
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
from .util import NATURAL, is_ipython
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import lru_cache, reduce, total_ordering
from math import pi as PI
from music21.midi.translate import streamToMidiFile
from mido import MidiFile
from numpy.typing import NDArray
from typing import Literal

_PITCH_NAME_REGEX = re.compile(r"[CDEFGAB](x?#+|b+)?(-?[0-9]+)")
PIANO_A0 = 21
PIANO_C8 = 108
_MUSIC21_SETUP = False

def _get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("You need to install sounddevice to use the play function")

def _setup():
    from music21 import environment
    global _MUSIC21_SETUP
    if _MUSIC21_SETUP:
        return

    us = environment.UserSettings()
    us['musescoreDirectPNGPath'] = '/usr/bin/mscore'
    us['directoryScratch'] = '/tmp'

    _MUSIC21_SETUP = True

class Audio:
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
            blocking = True # Otherwise jupyter notebook will behave weirdly
        else:
            if is_ipython():
                from IPython.display import Audio as IPAudio # type: ignore
                return IPAudio(self._data, rate = self.sample_rate)
            info = []
        info = sorted(info, key = lambda x: x[1])
        longest_info = max([len(x[0]) for x in info]) if info else 0
        sound = self._data.T
        self._thread = threading.Thread(target=_play, args=(sound, self.sample_rate, self.nchannels, lambda :self._stop_audio))
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
        self._stop_audio = False # Reset the state

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
        except (ValueError, RuntimeError) as e: # Seems like torchaudio changed the error type to runtime error in 2.2?
            # or the file path is invalid
            raise RuntimeError(f"Error saving the audio: {e} - {fpath}")

    def __repr__(self):
        """
        Prints out the following information about the audio:
        Duration, Sample rate, Num channels, Num frames
        """
        return f"(Audio)\nDuration:\t{self.duration:5f}\nSample Rate:\t{self.sample_rate}\nChannels:\t{self.nchannels}\nNum frames:\t{self.nframes}"


@dataclass(frozen=True, eq=True)
@total_ordering
class Note:
    """A piano note is a representation of a note on the piano, with a note name and an octave
    The convention being middle C is C4. The lowest note is A0 and the highest note is C8.

    If the note is time agnostic, then the duration and offset is timed with respect to quarter length,
    otherwise it is timed with respect to real-time seconds."""
    index: int
    octave: int
    duration: float
    offset: float
    time_agnostic: bool = False

    def __post_init__(self):
        # Sanity Check
        assert PIANO_A0 <= self.midi_number <= PIANO_C8, f"Note must be between A0 and C8, but found {self.midi_number}"
        assert self.duration >= 0, f"Duration must be greater than or equal to 0, but found {self.duration}"
        assert self.offset >= 0, f"Offset must be greater than or equal to 0, but found {self.offset}"

    def __repr__(self):
        return f"PianoNote({self.note_name}{self.octave})"

    @property
    def pitch_name(self) -> str:
        """Returns a note name of the pitch."""
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
        """The note name of the note."""
        return f"{self.pitch_name}{self.octave}"

    @property
    def step(self) -> Literal["C", "D", "E", "F", "G", "A", "B"]:
        """Returns the diatonic step of the note"""
        idx = self.index % 7
        return ("C", "G", "D", "A", "E", "B", "F")[idx]

    @property
    def alter(self):
        """Returns the alteration of the note aka number of sharps. Flats are represented as negative numbers."""
        return (self.index + 1) // 7

    @property
    def pitch_number(self):
        """Returns the chromatic pitch number of the note. C is 0, D is 2, etc"""
        return ([0, 2, 4, 5, 7, 9, 11][self.step_number] + self.alter) % 12

    @property
    def midi_number(self):
        """The chromatic pitch number of the note, using the convention that A4=440Hz converts to 69
        This is also the MIDI number of the note."""
        return self.pitch_number + 12 * self.octave + 12

    @property
    def step_number(self):
        """The step number of the note. A0 is 0, Middle C is 23 and in/decreases by 1 for each step."""
        return 7 * self.octave + self.step_number - 5

    @classmethod
    def from_str(cls, note: str, time_agnostic: bool = False) -> Note:
        """Creates a Note from a string note.

        Example: A4[0, 1] is A in the 4th octave with a duration of 0 and offset of 1."""
        duration = float(note.split("[")[1].split(",")[0])
        offset = float(note.split(",")[1].split("]")[0])
        note = note.split("[")[0]
        match = _PITCH_NAME_REGEX.match(note)
        assert match, f"Note {note} is not a valid note name"
        pitch_name, alter, octave = match.groups()
        alter = alter.replace("x", "##").replace("-", "b").replace("+", "#")
        sharps = reduce(lambda x, y: x + 1 if y == "#" else x - 1, alter, 0)
        assert pitch_name in ("C", "D", "E", "F", "G", "A", "B"), f"Step must be one of CDEFGAB, but found {pitch_name}" # to pass the typechecker
        return cls(
            index=step_alter_to_lof_index(pitch_name, sharps),
            octave=int(octave),
            duration=duration,
            offset=offset,
            time_agnostic=time_agnostic
        )

    @classmethod
    def from_note(cls, note: m21.note.Note, time_agnostic: bool = False) -> Note:
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
            step_alter_to_lof_index(step, int(alter)),
            octave,
            float(duration),
            float(offset),
            time_agnostic
        )

    @classmethod
    def from_midi_number(cls, midi_number: int, duration: float = 0., offset: float = 0., time_agnostic: bool = False) -> Note:
        """Creates a Note from a MIDI number. A4 maps to 69. If accidentals are needed, assumes the note is sharp."""
        octave = (midi_number // 12) - 1
        pitch = [0, 7, 2, 9, 4, -1, 6, 1, 8, 3, 10, 5][midi_number % 12]
        return cls(
            pitch,
            octave,
            duration,
            offset,
            time_agnostic
        )

    def __lt__(self, other: Note | int) -> bool:
        if isinstance(other, int):
            return self.midi_number < other
        return self.midi_number < other.midi_number

def step_alter_to_lof_index(step: Literal["C", "D", "E", "F", "G", "A", "B"], alter: int) -> int:
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter

def _convert_midi_to_wav(input_path: str, output_path: str, soundfont_path="~/.fluidsynth/default_sound_font.sf2", sample_rate=44100, verbose=False):
    subprocess.call(['fluidsynth', '-ni', soundfont_path, input_path, '-F', output_path, '-r', str(sample_rate)],
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.DEVNULL if not verbose else None)

def is_package_installed(package_name):
    try:
        result = subprocess.run(['dpkg', '-s', package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if 'install ok installed' in result.stdout:
            return True
    except subprocess.CalledProcessError:
        return False
    return False

def midi_to_notes(midi_path: str, time_agnostic: bool = False) -> list[Note]:
    if time_agnostic:
        # Use music21 to convert the midi to notes
        if not _MUSIC21_SETUP:
            _setup()
        stream = m21.converter.parse(midi_path)
        notes: list[Note] = []
        if not isinstance(stream, m21.stream.Score):
            raise ValueError(f"Midi file must contain a score, found {type(stream)}")
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
                note = Note.from_note(el, time_agnostic=True)
                # Use some python black magic to ensure the offset is calculated correctly
                object.__setattr__(note, "offset", offset)
                notes.append(note)
            elif isinstance(el, m21.chord.Chord):
                for el_ in el.notes:
                    note = Note.from_note(el_, time_agnostic=True)
                    # Use some python black magic to ensure the offset is calculated correctly
                    object.__setattr__(note, "offset", offset)
                    notes.append(note)
        notes = sorted(notes, key=lambda x: x.offset)
    else:
        mid = MidiFile(midi_path)
        tempo = 500000  # default tempo (microseconds per beat)
        ticks_per_beat = mid.ticks_per_beat

        current_time = 0  # current time in seconds
        notes = []
        note_on_dict = {}

        for track in mid.tracks:
            for msg in track:
                current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)

                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    note_on_dict[msg.note] = (current_time, msg.velocity)
                elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_on_dict:
                    start_time, _ = note_on_dict.pop(msg.note)
                    duration = current_time - start_time
                    note = Note.from_midi_number(midi_number=msg.note, duration=duration, offset=start_time, time_agnostic=False)
                    notes.append(note)
    assert all(note.time_agnostic == time_agnostic for note in notes)
    return notes

def score_to_audio(score: m21.stream.Score, sample_rate: int = 44100, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2") -> Audio:
    """Inner helper function to convert a music21 score to audio. The score will be consumed."""
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
    assert is_package_installed("fluidsynth"), "You need to install fluidsynth to convert midi to audio, refer to README for more details"
    if not _MUSIC21_SETUP:
        _setup()
    stream = m21.converter.parse(midi_path)
    if not isinstance(stream, m21.stream.Score):
        raise ValueError(f"Midi file must contain a score, found {type(stream)}")
    for n in stream.recurse().getElementsByClass((m21.note.Note, m21.chord.Chord, m21.dynamics.Dynamic)):
        if isinstance(n, (m21.note.Note, m21.chord.Chord)):
            n.volume = 0.5
        # This code fixes some weird dynamic problem in some edge cases but makes most music a little worse.
        # See if this is needed in the future
        # elif isinstance(n, m21.dynamics.Dynamic) and n.activeSite is not None:
        #     n.activeSite.remove(n)
    return score_to_audio(stream, sample_rate, soundfont_path)

def notes_to_score(notes: list[Note]) -> m21.stream.Score:
    """Convert a list of notes to a music21 score. The score is only intended to be played and not for further analysis."""
    assert all(note.time_agnostic for note in notes), "All notes must be time agnostic"
    if not _MUSIC21_SETUP:
        _setup()

    score = m21.stream.Score()

    notes = sorted(notes, key=lambda x: x.offset)
    heap = []

    # This dictionary will store the clef assignments (clef number -> list of events)
    assignments: dict[int, list[Note]] = {}
    clef_ctr = 0

    # Iterate through the sorted list of events
    for note in notes:
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
            measure.append(m21note)
        part.append(measure)
        score.append(part)
        part.makeRests(inPlace=True, fillGaps=True)
    return score

def notes_to_audio(notes: list[Note], sample_rate: int = 44100, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2") -> Audio:
    """Converts a list of notes to audio. This function will convert the notes to a midi file and then to audio."""
    score = notes_to_score(notes)
    return score_to_audio(score, sample_rate, soundfont_path)
