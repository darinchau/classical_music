from __future__ import annotations
from ..util import is_ipython

import os
import tempfile
import threading
from numpy.typing import NDArray
import typing
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def _get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("You need to install sounddevice to use the play function")


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
