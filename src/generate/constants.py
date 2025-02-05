import math


def sign(x):
    return math.copysign(1, x)


# Names and ranges
KEY_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
KEY_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E',
                   'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SHARPS_IDX = [1, 3, 6, 8, 10]

BASS_RANGE = list(range(40, 65))
TENOR_RANGE = list(range(48, 73))
ALTO_RANGE = list(range(53, 78))
SOPRANO_RANGE = list(range(60, 85))

BASS = 0
TENOR = 1
ALTO = 2
SOPRANO = 3

RANGES = [BASS_RANGE, TENOR_RANGE, ALTO_RANGE, SOPRANO_RANGE]

# intervals
P1 = C = Tonic = Unison = 0
m2 = Db = 1
M2 = D = 2
m3 = Eb = 3
M3 = E = 4
P4 = F = 5
d5 = Gb = Tritone = 6
P5 = G = 7
m6 = Ab = 8
M6 = A = 9
m7 = Bb = 10
M7 = B = 11
P8 = Octave = 12

NAMED_SCALES = {
    "major": (2, 2, 1, 2, 2, 2, 1),
    "minor": (2, 1, 2, 2, 1, 2, 2),
}

SPECIES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
}

RANGES_NAMES = {
    "bass": 0,
    "tenor": 1,
    "alto": 2,
    "soprano": 3
}

MELODIC_CONSONANT_INTERVALS = [m2, M2, m3, M3, P4, P5, m6, Octave]
MELODIC_INTERVALS = [Unison, m2, M2, m3, M3, P4, P5, m6, P8, -m2, -M2, -m3, -M3, -P4, -P5, -P8]
HARMONIC_DISSONANT_INTERVALS = [m2, M2, P4, M7, m7, P8 + m2, P8 + M2]
HARMONIC_CONSONANCES = [m3, M3, P5, m6, M6, P8, P8 + m3, P8 + M3]
PERFECT_INTERVALS = [P5, P8]
