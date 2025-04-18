import sys
import os


class NotSupportedOnWindows(NotImplementedError):
    pass


def is_ipython():
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False


_music21_setup = False


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


def _require_music21():
    """Check if music21 is installed and set up."""
    global _music21_setup
    if not _music21_setup:
        try:
            from music21 import environment
        except ImportError:
            raise ImportError("Music21 is not installed. Please install it using `pip install music21`.")
        _setup()
        _music21_setup = True


class _shutup:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


NATURAL = "♮"
PIANO_A0 = 21
PIANO_C8 = 108
