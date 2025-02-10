# Contains all auxiliary functions used in modelling stuff and generating/processing data samples
import librosa
from math import ceil
import numpy as np
from ..data import Note, notes_to_pianoroll, notes_to_audio


def notes_to_representations(notes: list[Note], n_fft: int = 512):
    pianoroll = notes_to_pianoroll(notes)

    audio = notes_to_audio(notes)
    audio_implied_duation = np.max(np.where(audio.data > 0)[1]) / audio.sample_rate

    # Use mono audio for now, maybe TODO handle stereo
    audio = audio.to_nchannels(1)

    hop_length = audio.sample_rate / pianoroll.resolution
    if not hop_length.is_integer():
        raise ValueError(f"Sample rate ({audio.sample_rate}) must be a multiple of the resolution ({pianoroll.resolution})")
    hop_length = int(hop_length)

    # Process the piano roll
    pianoroll_frames = int(audio_implied_duation * pianoroll.resolution) + 1
    roll = np.pad(pianoroll._pianoroll, ((0, pianoroll_frames - pianoroll._pianoroll.shape[0] + 1), (0, 0)), mode="constant")

    target = ceil(pianoroll_frames * int(audio.sample_rate / pianoroll.resolution) / hop_length) * hop_length + n_fft
    audio = audio.pad(target)

    spec = librosa.stft(audio.data, n_fft=n_fft, hop_length=hop_length, window='hann')
    spec = spec[0].T

    # Stack the magnitude and phase
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    x = np.stack([magnitude, sin_phase, cos_phase], axis=-1)
    return (
        # Shape: T, int(n_fft/2) + 1, 3
        x,
        # Shape: T, 90,
        roll,
    )
