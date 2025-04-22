# This module implements all the dumb methods

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
import shutil
import subprocess
import tempfile
import threading
import typing
import warnings
from ..util import NATURAL, is_ipython, _require_music21
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
from music21.stream.base import Score, Stream, Part, Measure
from mido import MidiFile, MidiTrack, Message
from numpy.typing import NDArray
from typing import Literal
from .base import SymbolicMusic
from .audio import Audio

if typing.TYPE_CHECKING:
    from .notes import Note, RealTimeNotes, NotatedTimeNotes


class XmlFile(SymbolicMusic):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    @classmethod
    def load_from_xml(cls, path: str):
        """Loads a MusicXML file."""
        assert os.path.exists(path), f"File {path} does not exist"
        return cls(path)

    def to_score(self) -> Music21Stream:
        """Loads a MusicXML file and returns a music21 stream."""
        _require_music21()
        stream = m21.converter.parse(self.path)
        if not isinstance(stream, Stream):
            raise ValueError(f"Stream must be a music21 stream, found {type(stream)}")
        return Music21Stream(stream)

    def save_to_midi(self, path: str):
        """Saves the MusicXML file to a MIDI file."""
        return self.to_score().save_to_midi(path)


class Midifile(SymbolicMusic):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @property
    def path(self) -> str:
        """Returns the path to the MIDI file."""
        return self._path

    @classmethod
    def load_from_xml(cls, path: str):
        """Loads a MIDI file from a MusicXML file."""
        _require_music21()
        with tempfile.TemporaryDirectory(delete=False) as tmpdir:
            p = os.path.join(tmpdir, os.path.basename(path) + ".mid")
            XmlFile.load_from_xml(path).to_score().save_to_midi(p)
        return cls(p)

    def save_to_midi(self, path: str):
        """Saves the MIDI file to a new path."""
        shutil.copy(self._path, path)

    def to_notated_time_notes(self) -> NotatedTimeNotes:
        _require_music21()
        stream = m21.converter.parse(self.path)
        if not isinstance(stream, Score):
            raise ValueError(f"Midi file must contain a score, found {type(stream)}")
        return Music21Stream(stream).to_notated_time_notes()

    def to_real_time_notes(self) -> RealTimeNotes:
        from .notes import Note, RealTimeNotes, NotatedTimeNotes
        mid = mido.MidiFile(self.path)
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

    def to_audio(self, sample_rate: int = 44100, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2") -> Audio:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            _convert_midi_to_wav(self.path, f.name, soundfont_path, sample_rate)
            return Audio.load(f.name)


class Music21Stream(SymbolicMusic):
    def __init__(self, stream: Stream):
        super().__init__()
        self._stream = stream

    def __getitem__(self, item):
        return self._stream[item]

    @classmethod
    def load_from_xml(cls, path: str):
        """Loads a music21 stream from a MusicXML file."""
        assert os.path.exists(path), f"File {path} does not exist"
        _require_music21()
        stream = m21.converter.parse(path)
        if not isinstance(stream, Stream):
            raise ValueError(f"Stream must be a music21 stream, found {type(stream)}")
        return cls(stream)

    @classmethod
    def load_from_midi(cls, path: str):
        """Loads a music21 stream from a MIDI file."""
        return cls.load_from_xml(path)  # Should work???

    def save_to_midi(self, path: str):
        """Saves the music21 stream to a MIDI file."""
        _require_music21()
        midi_file = streamToMidiFile(self._stream)
        midi_file.open(path, "wb")
        try:
            midi_file.write()
        finally:
            midi_file.close()

    def get_stream(self):
        return copy.deepcopy(self._stream)

    def to_notated_time_notes(self):
        from .notes import Note, RealTimeNotes, NotatedTimeNotes
        notes: list[Note] = []
        for el in self._stream.recurse().getElementsByClass((
            m21.note.Note,
            m21.chord.Chord,
        )):
            # Calculate the offset
            x: m21.base.Music21Object = el
            offset = Fraction()
            while x.activeSite is not None:
                offset += x.offset
                x = x.activeSite
                if x.id == self._stream.id:
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

    def show(self):
        return display_score(self._stream)

    def to_audio(self, sample_rate: int = 44100, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2") -> Audio:
        _require_music21()
        with (
            tempfile.NamedTemporaryFile(suffix=".mid") as f1,
        ):
            self.save_to_midi(f1.name)
            return Midifile(f1.name).to_audio(sample_rate, soundfont_path)


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


def display_score(obj: m21.base.Music21Object, invert_color: bool = True, skip_display: bool = False):
    """Displays the score. Returns a dictionary where keys are the page numbers and values are the images of the page in np arrays"""
    _require_music21()

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
