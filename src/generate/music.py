from __future__ import annotations
from .constants import *
import math
import random as rm
from ..data import PIANO_A0, PIANO_C8, Note


def _make_note(pitch: int | str):
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
    is not nessecarily the lowest note.
    """

    def __init__(self, key, scale, scale_range=None):
        if key[0].upper() not in (KEY_NAMES_SHARP or KEY_NAMES):
            print("Error, key name not valid. Try on the format 'C' or 'Db' ")
            pass
        if key in ["A", "A#", "B", "Bb"]:
            oct = 0
        else:
            oct = 1
        self.root = _make_note(key + str(oct))  # sets the root of the scale in valid string format
        self.key = key
        self.scale_type = scale
        if isinstance(scale, str):
            scale = Scale.intervals_from_name(scale)
        elif isinstance(scale, Scale):
            scale = scale.intervals
        self.intervals = tuple(scale)
        self.scale = self.build_scale()
        self.scale_pitches = self.get_scale_pitches()
        if scale_range != None:
            self.limit_range(scale_range)

    @staticmethod
    def intervals_from_name(scale_name):
        global NAMED_SCALES
        scale_name = scale_name.lower()

        # supporting alternative formatting..
        for text in ['scale', 'mode']:
            scale_name = scale_name.replace(text, '')
        for text in [" ", "-"]:
            scale_name = scale_name.replace(text, "_")
        return NAMED_SCALES[scale_name]

    def build_scale(self):
        start_pitch = self.root.midi_number
        scale_len = len(self.intervals)
        highest_possible_pitch = PIANO_C8
        lowest_possible_pitch = PIANO_A0
        j = 0
        scale = []
        pitch = start_pitch
        # adds all possible values above the root pitch
        while pitch <= highest_possible_pitch:
            scale.append(_make_note(pitch))
            pitch = scale[j].get_pitch() + self.intervals[j % scale_len]
            j += 1
        # adds all possible values under the root pitch
        j = scale_len - 1
        pitch = start_pitch - self.intervals[j % scale_len]
        while pitch >= lowest_possible_pitch:
            scale.insert(0, _make_note(pitch))
            j -= 1
            pitch = pitch - self.intervals[j % scale_len]
        return scale

    def get_scale_pitches(self):
        scale_pitches = []
        for notes in self.scale:
            scale_pitches.append(notes.get_pitch())
        return scale_pitches

    def get_scale_range(self, scale_range):
        """
              :param scale_range: [int] list of note pitches in the range to be returned
              :return: the scale limited to the given range
              """
        scale_pitches = []
        for notes in scale_range:
            if notes in self.scale_pitches:
                scale_pitches.append(notes)
        return scale_pitches

    def limit_range(self, scale_range):
        scale = []
        for notes in scale_range:
            if notes in self.scale_pitches:
                scale.append(_make_note(notes))
        self.scale = scale

    def set_time(self, duration):
        t = 0
        for notes in self.scale:
            notes.set_time(t, duration)
            t += duration


class Melody:
    def __init__(self, key, scale, bar_length, melody_notes=None, melody_rhythm=None, ties=None, start=0, voice_range=None):
        "Search Space"
        self.key = key
        self.scale_name = scale
        self.voice_range = voice_range
        self.scale = Scale(key, scale, voice_range)
        self.scale_pitches = self.scale.get_scale_pitches()
        self.note_resolution = 8
        self.start = start
        self.bar_length = float(bar_length)

        """Music Representation"""
        self.pitches = melody_notes
        self.rhythm = melody_rhythm
        self.ties = ties
        if self.pitches != None:
            self.search_domain = [self.scale_pitches for notes in self.pitches]
        else:
            self.search_domain = [self.scale_pitches]

    def set_ties(self, ties):
        self.ties = ties.copy()

    def set_rhythm(self, rhythm):
        self.rhythm = rhythm.copy()

    def set_melody(self, melody):
        self.pitches = melody.copy()

    def get_ties(self):
        return self.ties.copy()

    def get_rhythm(self):
        return self.rhythm.copy()

    def get_melody(self):
        return self.pitches.copy()

    def get_end_time(self):
        t = self.start
        for elem in self.rhythm:
            t += elem
        return t*self.bar_length / float(self.note_resolution)
