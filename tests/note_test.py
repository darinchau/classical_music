import pytest
from src.data import Note


def test_note_from_str():
    assert Note.from_str("C4[0, 1, 64]").midi_number == 60
    assert Note.from_str("C4[0, 1, 64]").duration == 0
    assert Note.from_str("C4[0, 1, 64]").velocity == 64
    assert Note.from_str("C4[0, 1, 64]").offset == 1

    assert Note.from_str("G4[2, 0, 80]").midi_number == 67
    assert Note.from_str("G4[2, 0, 80]").duration == 2
    assert Note.from_str("G4[2, 0, 80]").velocity == 80
    assert Note.from_str("G4[2, 0, 80]").offset == 0

    with pytest.raises(AssertionError):
        Note.from_str("C4[0, -1, 64")
    with pytest.raises(AssertionError):
        Note.from_str("C4[-1, 1, 64")
    with pytest.raises(AssertionError):
        Note.from_str("C4[0, 1, 128]")
    with pytest.raises(AssertionError):
        Note.from_str("C4[0, 1, -1]")
    with pytest.raises(AssertionError):
        Note.from_str("C4[0]")
    with pytest.raises(AssertionError):
        Note.from_str("C4[]")

    assert Note.from_str("G4[0, 2]").midi_number == 67
    assert Note.from_str("G4[0, 2]").duration == 0
    assert Note.from_str("G4[0, 2]").offset == 2

    assert Note.from_str("C4").midi_number == 60
    assert Note.from_str("A4").midi_number == 69
    assert Note.from_str("D#3").midi_number == 51
    assert Note.from_str("Fb4").midi_number == 64
    assert Note.from_str("E#4").midi_number == 65
    assert Note.from_str("B#3").midi_number == 60
    assert Note.from_str("Cb4").midi_number == 59
    assert Note.from_str("A0").midi_number == 21
    assert Note.from_str("C8").midi_number == 108
    assert Note.from_str("B##4").midi_number == 73
    assert Note.from_str("G##4").midi_number == 69
    assert Note.from_str("Gbb4").midi_number == 65

    with pytest.raises(AssertionError):
        Note.from_str("H4")
    with pytest.raises(AssertionError):
        Note.from_str("C#10")
    with pytest.raises(AssertionError):
        Note.from_str("C-1")
    with pytest.raises(AssertionError):
        Note.from_str("4C")

    # Implied octave is 4
    assert Note.from_str("C").midi_number == 60
