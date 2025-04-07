from .data import Audio, RealTimeNotes
from abc import ABC, abstractmethod
import tempfile
from .data import notes_to_midi, midi_to_audio

class NotesPlayer(ABC):
    """This will be used to play notes into audio. We have no way of checking this
    but ideally the notes being played has timing aligned exactly with the audio
    This should only accept notes that are real-timed
    In the future we could hook this to a VST

    A simple example of this is the default fluid synth player implemented below"""
    @abstractmethod
    def play(self, notes: RealTimeNotes, sample_rate: int = 48000) -> Audio:
        raise NotImplementedError


class FluidSynthNotesPlayer(NotesPlayer):
    """Uses the fluidsynth library to play notes into audio"""

    def __init__(self, soundfont_path: str = "~/.fluidsynth/default_sound_font.sf2"):
        self._soundfont_path = soundfont_path

    def play(self, notes: RealTimeNotes, sample_rate: int = 48000) -> Audio:
        """Plays the notes into audio"""
        with tempfile.NamedTemporaryFile(suffix=".mid") as f:
            notes_to_midi(notes, f.name)
            return midi_to_audio(f.name, sample_rate=sample_rate, soundfont_path=self._soundfont_path)
