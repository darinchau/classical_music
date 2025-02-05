import itertools
import math
import random as rm
import time
from . import music as m
from .constants import *
from .constraints import Constraints


class CantusFirmus(m.Melody):
    # Some constants for easy access
    perfect_intervals = [Unison, P5, Octave]
    dissonant_intervals = [m7, M7, Tritone, -m6, -m7, -M7]
    consonant_melodic_intervals = [m2, M2, m3, M3, P4, P5, m6, Octave, -m2, -M2, -m3, -M3, -P4, -P5, -Octave]

    def __init__(self, key, scale, bar_length, melody_notes=None, melody_rhythm=None, start=0, voice_range=RANGES[ALTO]):
        super(CantusFirmus, self).__init__(key, scale, bar_length, melody_notes=melody_notes, melody_rhythm=melody_rhythm,
                                           start=start, voice_range=voice_range)
        self.cf_errors = []

        """ Music representation"""
        self.rhythm = self._generate_rhythm()
        self.ties = [False]*len(self.rhythm)
        self.pitches = self._generate_cf()
        self.length = len(self.rhythm)
        self.ERROR_THRESHOLD: int = 0

    def _start_note(self):
        root = self.key
        try:
            root_idx = KEY_NAMES.index(root)
        except:
            root_idx = KEY_NAMES_SHARP.index(root)
        v_range = self.voice_range
        possible_start_notes = []
        for pitches in v_range:
            if pitches % Octave == root_idx:
                possible_start_notes.append(pitches)
        tonics = possible_start_notes
        return tonics, possible_start_notes[0]

    def _penultimate_note(self):
        """ The last note can be approached from above or below.
            It is however most common that the last note is approached from above
        """
        leading_tone = self._start_note()[1] - 1
        super_tonic = self._start_note()[1] + 2
        weights = [0.1, 0.9]  # it is more common that the penultimate note is the supertonic than leading tone
        penultimate_note = rm.choices([leading_tone, super_tonic], weights)[0]
        return penultimate_note

    def _get_leading_tones(self):
        if self.scale_name == "minor":
            leading_tone = self._start_note()[1] - 2
        else:
            leading_tone = self._start_note()[1] - 1
        return [leading_tone]

    def _generate_rhythm(self):
        """
        Generates the number of bars for the cantus firmus. Length between 8 and 13 bars, with 12 being most common.
        Therefore modelled as a uniform distribution over 8 to 14
        :return:
        """
        random_length = rm.randint(8, 14)
        return [(8,)]*random_length

    def _is_step(self, note, prev_note):
        if abs(prev_note-note) in [m2, M2]:
            return True
        else:
            return False

    def _is_small_leap(self, note, prev_note):
        if abs(prev_note-note) in [m3, M3]:
            return True
        else:
            return False

    def _is_large_leap(self, note, prev_note):
        if abs(prev_note-note) >= P4:
            return True
        else:
            return False

    """ RULES FOR THE CANTUS FIRMUS"""

    def _is_climax(self, cf_shell):
        if cf_shell.count(max(cf_shell)) == 1:
            return True
        else:
            return False

    def _is_resolved_leading_tone(self, cf_shell):
        tonics = self._start_note()[0]
        leading_tones = self._get_leading_tones()
        for leading_tone in leading_tones:
            if leading_tone in cf_shell and cf_shell[cf_shell.index(leading_tone)+1] != tonics[0]:
                return False
        return True

    def _is_dissonant_intervals(self, cf_shell):
        for i in range(len(cf_shell)-1):
            if cf_shell[i+1] - cf_shell[i] in self.dissonant_intervals:
                return True
        return False

    def _check_leaps(self, cf_shell):
        penalty = 0
        num_large_leaps = 0
        for i in range(len(cf_shell)-2):
            if self._is_large_leap(cf_shell[i], cf_shell[i+1]):
                num_large_leaps += 1
                if abs(cf_shell[i]-cf_shell[i+1]) == Octave:
                    # small penalty for octave leap
                    self.cf_errors.append("penalty for octave leap")
                    penalty += 50
                # Check consecutive leaps first
                elif self._is_large_leap(cf_shell[i+1], cf_shell[i+2]):
                    self.cf_errors.append("consecutive leaps")
                    penalty += 25
                elif self._is_large_leap(cf_shell[i+1], cf_shell[i+2]) and sign(cf_shell[i+1]-cf_shell[i]) != sign(cf_shell[i+2]-cf_shell[i+1]):
                    self.cf_errors.append("Large leaps in opposite direction")
                    penalty += 75
                elif self._is_step(cf_shell[i+1], cf_shell[i+2]) and sign(cf_shell[i+1]-cf_shell[i]) == sign(cf_shell[i+2]-cf_shell[i+1]):
                    self.cf_errors.append("A leap is not properly recovered")
                    penalty += 75
        if num_large_leaps >= int(len(self.rhythm) / 2) - 2:
            penalty += 100
        return penalty

    def _is_valid_note_count(self, cf_shell):
        for notes in set(cf_shell):
            if cf_shell.count(notes) > 4:
                return False
            else:
                return True

    def _is_valid_range(self, cf_shell):
        if abs(max(cf_shell)-min(cf_shell)) > Octave+M3:
            return False
        else:
            return True

    def _is_repeated_motifs(self, cf_shell):
        paired_notes = []
        for i in range(len(cf_shell)-1):
            if cf_shell[i] == cf_shell[i+1]:
                return True
            if cf_shell[i] == cf_shell[0] and i != 0:
                return True
            paired_notes.append([cf_shell[i], cf_shell[i+1]])
        for pairs in paired_notes:
            if paired_notes.count(pairs) > 1:
                return True
        return False

    def _cost_function(self, cf_shell):
        penalty = 0
        penalty = self._check_leaps(cf_shell)
        if not self._is_valid_note_count(cf_shell):
            self.cf_errors.append("note repetition")
            penalty += 100
        if not self._is_climax(cf_shell):
            self.cf_errors.append("no unique cf climax")
            penalty += 100
        if not self._is_valid_range(cf_shell):
            self.cf_errors.append("exceeds the range of a tenth")
            penalty += 100
        if self._is_repeated_motifs(cf_shell):
            self.cf_errors.append("motivic repetitions")
            penalty += 100
        if not self._is_resolved_leading_tone(cf_shell):
            self.cf_errors.append("leading tone not resolved")
            penalty += 100
        if self._is_dissonant_intervals(cf_shell):
            self.cf_errors.append("dissonant interval")
            penalty += 100
        return penalty

    def _initialize_cf(self):
        """
        Randomizes the initial cf and sets correct start, end, and penultimate notes.
        :return: list of cf pitches.
        """
        start_note = self._start_note()[1]
        end_note = start_note
        penultimate_note = self._penultimate_note()
        length = len(self.rhythm)
        cf_shell = [rm.choice(self.scale_pitches) for i in range(length)]
        cf_shell[0] = start_note
        cf_shell[-1] = end_note
        cf_shell[-2] = penultimate_note
        return cf_shell

    def _get_melodic_consonances(self, prev_note):
        mel_cons = []
        for intervals in self.consonant_melodic_intervals:
            if prev_note+intervals in self.scale_pitches:
                mel_cons.append(prev_note+intervals)
        # To further randomize the generated results, the melodic consonances are shuffled
        rm.shuffle(mel_cons)
        return mel_cons

    def _generate_cf(self):
        total_penalty = math.inf
        iteration = 0
        while total_penalty > 0:
            cf_shell = self._initialize_cf()  # initialized randomly
            for i in range(1, len(cf_shell)-2):
                self.cf_errors = []
                local_max = math.inf
                cf_draft = cf_shell.copy()
                mel_cons = self._get_melodic_consonances(cf_shell[i-1])
                for notes in mel_cons:
                    cf_draft[i] = notes
                    local_penalty = self._cost_function(cf_draft)
                    if local_penalty <= local_max:
                        local_max = local_penalty
                        best_choice = notes
                cf_shell[i] = best_choice
            self.cf_errors = []
            total_penalty = self._cost_function(cf_shell)
            iteration += 1
        return cf_shell.copy()


class Counterpoint:
    def __init__(self, cf: CantusFirmus, ctp_position="above"):
        if ctp_position == "above":
            self.voice_range = RANGES[RANGES.index(cf.voice_range)+1]
        else:
            self.voice_range = RANGES[RANGES.index(cf.voice_range)-1]
        self.melody = m.Melody(cf.key, cf.scale, cf.bar_length, voice_range=self.voice_range)
        self.ctp_position = ctp_position
        self.scale_pitches = self.melody.scale_pitches
        self.cf = cf
        self.species = None
        self.search_domain = []
        self.ctp_errors = []

    @property
    def MAX_SEARCH_TIME(self):
        return 5

    """ VALID START, END, AND PENULTIMATE NOTES"""

    def _start_notes(self):
        cf_tonic = self.cf.pitches[0]
        if self.ctp_position == "above":
            if SPECIES[self.species] == 1:
                return [cf_tonic, cf_tonic + P5, cf_tonic + Octave]
            else:
                return [cf_tonic+P5, cf_tonic + Octave]
        else:
            if SPECIES[self.species] == 1:
                return [cf_tonic - Octave, cf_tonic]
            else:
                return [cf_tonic - Octave]

    def _end_notes(self):
        cf_tonic = self.cf.pitches[0]
        if self.ctp_position == "above":
            return [cf_tonic, cf_tonic + Octave]
        else:
            return [cf_tonic, cf_tonic - Octave]

    def _penultimate_notes(self, cf_end):
        cf_direction = [sign(self.cf.pitches[i] - self.cf.pitches[i - 1]) for i in range(1, len(self.cf.pitches))]
        if self.ctp_position == "above":
            s = 1
        else:
            s = -1
        if cf_direction[-1] == 1.0:
            penultimate = cf_end + 2
        else:
            penultimate = cf_end - 1
        return [penultimate, penultimate + s * Octave]

    """ INITIALIZING COUNTERPOINT WITH RANDOM VALUES"""

    def get_consonant_possibilities(self, cf_note):
        poss = []
        for interval in HARMONIC_CONSONANCES:
            if self.ctp_position == "above":
                if cf_note+interval in self.scale_pitches:
                    poss.append(cf_note+interval)
            else:
                if cf_note-interval in self.scale_pitches:
                    poss.append(cf_note-interval)
        return poss

    def randomize_ctp_melody(self):
        ctp_melody = []
        i = 0
        measure = 0
        while measure < len(self.melody.rhythm):
            note_duration = 0
            while note_duration < len(self.melody.rhythm[measure]):
                if i == 0:
                    ctp_melody.append(rm.choice(self.search_domain[i]))
                elif i > 0 and self.melody.ties[i-1] == True:
                    ctp_melody.append(ctp_melody[i-1])
                else:
                    ctp_melody.append(rm.choice(self.search_domain[i]))
                i += 1
                note_duration += 1
            measure += 1
        return ctp_melody

    """ GENERATE COUNTERPOINT PITCHES BY CALLING THE SEARCH ALGORITHM"""

    def generate_ctp(self):
        if self.species == None:
            print("No species to generate!")
        self.melody.set_melody(self.randomize_ctp_melody())
        self.ctp_errors = []
        self.error, best_ctp, self.ctp_errors = search.improved_search(self)
        self.melody.set_melody(best_ctp)


class FirstSpecies(Counterpoint):
    def __init__(self, cf, ctp_position="above"):
        super(FirstSpecies, self).__init__(cf, ctp_position)
        self.species = "first"
        self.ERROR_THRESHOLD = 50
        self.melody.set_rhythm(self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())
    """ RHYTHM """

    def get_rhythm(self):
        "Voices all move together in the same rhythm as the cantus firmus."
        return [(8,)]*self.cf.length

    def get_ties(self):
        return [False]*self.cf.length
    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def _possible_notes(self):
        poss = [None for elem in self.melody.rhythm]
        for i in range(len(self.melody.rhythm)):
            if i == 0:
                poss[i] = self._start_notes()
            elif i == len(self.melody.rhythm)-2:
                poss[i] = self._penultimate_notes(self.cf.pitches[i+1])
            elif i == len(self.melody.rhythm)-1:
                poss[i] = self._end_notes()
            else:
                poss[i] = self.get_consonant_possibilities(self.cf.pitches[i])
        return poss


class SecondSpecies(Counterpoint):
    def __init__(self, cf, ctp_position="above"):
        super(SecondSpecies, self).__init__(cf, ctp_position)
        self.species = "second"
        self.ERROR_THRESHOLD = 50
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())
    """ HELP FUNCTIONS"""

    def get_downbeats(self):
        indices = list(range(len(self.cf.pitches)))*2
        return indices[::2]

    def get_upbeats(self):
        indices = list(range(len(self.cf.pitches)))*2
        return indices[1::2]

    """ RHYTHMIC RULES """

    def get_rhythm(self):
        rhythm = [(4, 4)]*(len(self.cf.pitches)-1)
        rhythm.append((8,))
        return rhythm

    def get_ties(self):
        return [False]*self.num_notes

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, idx, cf_note):
        poss = super(SecondSpecies, self).get_consonant_possibilities(cf_note)
        upbeats = self.get_upbeats()
        if idx in upbeats:
            if idx != 1:
                for diss in HARMONIC_DISSONANT_INTERVALS:
                    if self.ctp_position == "above":
                        if cf_note+diss in self.scale_pitches:
                            poss.append(cf_note+diss)
                    else:
                        if cf_note-diss in self.scale_pitches:
                            poss.append(cf_note-diss)
        return poss

    def _possible_notes(self):
        poss = [None for elem in range(self.num_notes)]
        i = 0
        for m in range(len(self.get_rhythm())):
            for n in range(len(self.get_rhythm()[m])):
                if m == 0:
                    # First measure. start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.get_rhythm()) - 2 and n == 1:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.get_rhythm())-1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(i, self.cf.pitches[m])
                i += 1
        return poss


class ThirdSpecies(Counterpoint):
    def __init__(self, cf, ctp_position="above"):
        super(ThirdSpecies, self).__init__(cf, ctp_position)
        self.species = "third"
        self.ERROR_THRESHOLD = 100
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    """ HELP FUNCTIONS"""

    def get_downbeats(self):
        indices = list(range(len(self.cf.pitches)))*4
        return indices[::2]

    def get_upbeats(self):
        indices = list(range(len(self.cf.pitches)))*4
        return indices[1::2]

    """ RHYTHMIC RULES """

    def get_rhythm(self):
        rhythm = [(2, 2, 2, 2)] * (len(self.cf.pitches) - 1)
        rhythm.append((8,))
        return rhythm

    def get_ties(self):
        return [False] * self.num_notes

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, idx, cf_note):
        poss = super(ThirdSpecies, self).get_consonant_possibilities(cf_note)
        upbeats = self.get_upbeats()
        if idx in upbeats:
            if idx != 1:
                for diss in HARMONIC_DISSONANT_INTERVALS:
                    if self.ctp_position == "above":
                        if cf_note + diss in self.scale_pitches:
                            poss.append(cf_note + diss)
                    else:
                        if cf_note - diss in self.scale_pitches:
                            poss.append(cf_note - diss)
        return poss

    def _possible_notes(self):
        poss = [None for elem in range(self.num_notes)]
        i = 0
        for m in range(len(self.get_rhythm())):
            for n in range(len(self.get_rhythm()[m])):
                if m == 0 and n in [0, 1]:
                    # First measure, after rest start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.get_rhythm()) - 2 and n == 3:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.get_rhythm()) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(i, self.cf.pitches[m])
                i += 1
        return poss


class FourthSpecies(Counterpoint):
    def __init__(self, cf, ctp_position="above"):
        super(FourthSpecies, self).__init__(cf, ctp_position)
        self.species = "fourth"
        self.ERROR_THRESHOLD = 25
        self.melody.set_rhythm(self.get_rhythm())
        self.num_notes = sum(len(row) for row in self.get_rhythm())
        self.melody.set_ties(self.get_ties())
        self.search_domain = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    """ HELP FUNCTIONS"""

    """ RHYTHMIC RULES """

    def get_rhythm(self):
        rhythm = [(4, 4)] * (self.cf.length - 1)
        rhythm.append((8,))
        return rhythm

    def get_ties(self):
        ties = []
        for i in range(self.num_notes-2):
            if i % 2 == 0:
                ties.append(False)
            else:
                ties.append(True)
        ties.append(False)
        ties.append(False)
        return ties

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, idx, cf_note):
        poss = super(FourthSpecies, self).get_consonant_possibilities(cf_note)
        return poss

    def _possible_notes(self):
        poss = [None for elem in range(self.num_notes)]
        i = 0
        for m in range(len(self.get_rhythm())):
            for n in range(len(self.get_rhythm()[m])):
                if m == 0:
                    # First measure. start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.get_rhythm()) - 2 and n == 1:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.get_rhythm()) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(i, self.cf.pitches[m])
                i += 1
        return poss


class FifthSpecies(Counterpoint):
    def __init__(self, cf, ctp_position="above"):
        super(FifthSpecies, self).__init__(cf, ctp_position)
        self.species = "fifth"
        self.ERROR_THRESHOLD = 100
        self.melody.set_rhythm(self.get_rhythm())
        self.rhythm = self.melody.get_rhythm()
        self.num_notes = sum(len(row) for row in self.rhythm)
        self.melody.set_ties(self.get_ties())
        self.search_domain = self._possible_notes()
        self.melody.set_melody(self.randomize_ctp_melody())

    """ RHYTHMIC RULES """

    def get_rhythm(self):
        rhythm = []
        measure_rhythms = [(2, 2, 2, 2), (4, 2, 2), (2, 2, 4), (4, 4),
                           (2, 1, 1, 2, 2), (2, 1, 1, 4), (4, 2, 1, 1), (2, 2, 2, 1, 1), (2, 1, 1, 2, 2)]
        rhythmic_weights = [75, 75, 75, 75, 10, 5, 5, 5, 5]
        for measures in range(len(self.cf.pitches)-1):
            if measures == 0:
                rhythm.append((4, 4))
            else:
                rhythm.append(rm.choices(measure_rhythms, rhythmic_weights)[0])
        rhythm.append((8,))
        return rhythm

    def get_ties(self):
        rhythm = self.rhythm
        ties = []
        for m in range(len(rhythm)-1):
            for n in range(len(rhythm[m])):
                if m == 0 and n == 1:
                    ties.append(True)
                elif m > 0 and n == len(rhythm[m])-1:
                    if rhythm[m+1][0] == rhythm[m][n]/2:
                        ties.append(True)
                    else:
                        ties.append(False)
                else:
                    ties.append(False)
        ties.append(False)
        ties.append(False)
        return ties

    """ VOICE INDEPENDENCE RULES """
    """ MELODIC RULES """

    """ HELP FUNCTIONS FOR INITIALIZING COUNTERPOINT"""

    def get_harmonic_possibilities(self, m, n, cf_note):
        add_diss = False
        if self.rhythm[m][n] == 1:
            add_diss = True
        if sum(self.rhythm[m][:n]) in [2, 6]:
            add_diss = True
        poss = super(FifthSpecies, self).get_consonant_possibilities(cf_note)
        if add_diss:
            for diss in HARMONIC_DISSONANT_INTERVALS:
                if self.ctp_position == "above":
                    if cf_note + diss in self.scale_pitches:
                        poss.append(cf_note + diss)
                else:
                    if cf_note - diss in self.scale_pitches:
                        poss.append(cf_note - diss)
        return poss

    def _possible_notes(self):
        poss = [None for elem in range(self.num_notes)]
        i = 0
        for m in range(len(self.rhythm)):
            for n in range(len(self.rhythm[m])):
                if m == 0:
                    # First measure. start notes
                    if n == 0:
                        poss[i] = [-1]
                    else:
                        poss[i] = self._start_notes()
                elif m == len(self.rhythm) - 2 and n == len(self.rhythm[m])-1:
                    # penultimate note before last measure.
                    poss[i] = self._penultimate_notes(self.cf.pitches[-1])
                elif m == len(self.rhythm) - 1:
                    # Last measure
                    poss[i] = self._end_notes()
                else:
                    poss[i] = self.get_harmonic_possibilities(m, n, self.cf.pitches[m])
                i += 1
        return poss


def _get_indices(ctp_len, idx, n_window):
    s_w = []
    for i in range(n_window):
        if idx + i < ctp_len:
            s_w.append(idx + i)
        else:
            s_w.append(ctp_len - 1 - i)
    s_w.sort()
    return [s_w[0], s_w[-1]]


def _path_search(ctp, search_window):
    paths = []
    ctp_draft = ctp.ctp.melody.copy()
    poss = ctp.search_domain
    for i in itertools.product(*poss[search_window[0]:search_window[1]+1]):
        paths.append(list(i))
    ctp_draft[search_window[0]:search_window[1]+1] = paths[0]
    best_ctp = ctp_draft.copy()
    best_local_error = math.inf
    best_error_list = []
    for path in paths:
        ctp_draft[search_window[0]:search_window[1] + 1] = path
        ctp.ctp.set_melody(ctp_draft)
        enforced_constrains = Constraints(ctp)
        local_error = enforced_constrains.get_penalty()
        ctp_errors = enforced_constrains.get_errors()
        weighted_indices = enforced_constrains.get_weighted_indices()
        if local_error < best_local_error:
            best_ctp = ctp_draft.copy()
            ctp.ctp.set_melody(best_ctp)
            best_local_error = local_error
            best_error_list = ctp_errors
    return best_ctp.copy(), best_local_error, best_error_list


def search(ctp):
    error = math.inf
    best_scan_error = math.inf
    j = 1
    best_ctp = ctp.ctp.melody.copy()
    best_error_list = []
    while error >= ctp.ERROR_THRESHOLD and j <= ctp.MAX_SEARCH_WIDTH:
        error_window = math.inf
        for i in range(len(ctp.ctp.melody)):
            window_n = _get_indices(len(ctp.ctp.melody), i, 2)
            ctp_draft, error, list_of_errors = _path_search(ctp, window_n)
            if i == 0:
                error_window = error
            if error < best_scan_error:
                best_ctp = ctp_draft.copy()
                ctp.ctp.set_melody(best_ctp)
                best_scan_error = error
                best_error_list = list_of_errors
                if error < ctp.ERROR_THRESHOLD:
                    return best_scan_error, best_ctp, best_error_list

            """ steps_since_improvement += 1
            if steps_since_improvement >= len(ctp.ctp.melody):
                ctp.ctp.set_melody(ctp.randomize_ctp_melody())
                steps_since_improvement = 0
                j = 1"""
        ctp.ctp.set_melody(best_ctp)
        if error_window >= best_scan_error:
            j += 1
    return best_scan_error, best_ctp, best_error_list


def brute_force(ctp):
    penalty = math.inf
    errors = []
    while penalty > ctp.ERROR_THRESHOLD:
        ctp.ctp.set_melody(ctp.randomize_ctp_melody())
        enforced_constrains = Constraints(ctp)
        local_error = enforced_constrains.get_penalty()
        ctp_errors = enforced_constrains.get_errors()
        penalty = local_error
        errors = ctp_errors
    return penalty, ctp.ctp.melody, errors


def best_first_search(ctp, weighted_idx):
    search_domain = ctp.search_domain
    search_ctp = ctp.melody.get_melody()
    best_global_ctp = search_ctp.copy()
    best_global_error = math.inf
    best_global_weighted_indices = []
    if isinstance(weighted_idx, list):
        idx = weighted_idx
    else:
        idx = list(weighted_idx.keys())
    for i in idx:
        best_note = search_domain[i][0]
        local_error = math.inf
        local_weighted_indices = []
        for j in range(len(search_domain[i])):
            search_ctp[i] = search_domain[i][j]
            ctp.melody.set_melody(search_ctp.copy())
            constrained = Constraints(ctp)
            error = constrained.cost_function()
            weighted_indices = constrained.get_weighted_indices()
            if error <= local_error:
                best_note = search_domain[i][j]
                local_error = error
                local_weighted_indices = weighted_indices
        search_ctp[i] = best_note
        if local_error < best_global_error:
            best_global_ctp = search_ctp.copy()
            best_global_error = local_error
            best_global_weighted_indices = local_weighted_indices
    return best_global_error, best_global_ctp, best_global_weighted_indices


def improved_search(ctp: Counterpoint):
    start_time = time.time()
    penalty = math.inf
    elapsed_time = time.time()-start_time
    best_ctp = ctp.melody.get_melody()
    lowest_penalty = math.inf
    weighted_idx = [i for i in range(len(best_ctp))]
    prev_penalty = penalty
    randomize_idx = 1
    while penalty >= ctp.ERROR_THRESHOLD and elapsed_time < ctp.MAX_SEARCH_TIME:
        penalty, ctp_notes, weighted_idx = best_first_search(ctp, weighted_idx)
        if penalty == prev_penalty:  # no improvement
            weighted_idx = list(weighted_idx.keys())
            for i in range(randomize_idx):
                ctp_notes[weighted_idx[i]] = rm.choice(ctp.search_domain[weighted_idx[i]])
            rm.shuffle(weighted_idx)
            ctp.melody.set_melody(ctp_notes)
            if randomize_idx != len(best_ctp)-1:
                randomize_idx += 1
        if penalty < lowest_penalty:
            randomize_idx = 1
            best_ctp = ctp_notes
            ctp.melody.set_melody(best_ctp)
            lowest_penalty = penalty
            weighted_idx = weighted_idx
        elapsed_time = time.time()-start_time
        prev_penalty = penalty
    constraint = Constraints(ctp)
    lowest_penalty = constraint.cost_function()
    lowest_error_list = constraint.get_errors()
    return lowest_penalty, best_ctp, lowest_error_list
