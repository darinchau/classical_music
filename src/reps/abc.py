# Implements the ABC notation system
# Code partially borrowed from https://github.com/ElectricAlexis/NotaGen

from abctoolkit.transpose import Key2index, transpose_an_abc_text
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.rotate import rotate_abc
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.utils import (
    remove_information_field,
    remove_bar_no_annotations,
    Quote_re,
    Barlines,
    extract_metadata_and_parts,
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from tqdm import tqdm
import random
import json
import subprocess
import os
import sys
import tempfile
import shutil
import typing
import re
from .base import SymbolicMusic
from .data import XmlFile
from ..util import _shutup

_MAYBE_SUCCESSFUL = re.compile(r"written with [0-9]+ voices")


class ABCNotation(SymbolicMusic):
    """Implements the ABC notation for music where every line is separate by a new-line character"""

    def __init__(self, abc_string: str):
        self.abc_string = abc_string
        self._validate_abc()

    def _validate_abc(self):
        # TODO add validation
        pass

    @classmethod
    def load_from_xml(cls, path: str):
        """
        Load ABC notation from an XML file.
        :param path: Path to the XML file.
        :return: An instance of ABCNotation.
        """
        abc_string = xml2abc(path)
        abc_string = interleave_abc(abc_string)
        return cls("\n".join(abc_string))

    def save_to_midi(self, path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            abc_path = os.path.join(tmpdir, "temp.abc")
            with open(abc_path, 'w') as f:
                f.write(self.abc_string)
            xml_path = run_abc2xml([abc_path], output_directory=tmpdir)[0]
            if not os.path.exists(xml_path):
                raise ValueError(f"Failed to convert ABC to XML: {xml_path}")
            return XmlFile(xml_path).save_to_midi(path)


def interleave_abc(abc: str) -> list[str]:
    abc_lines = [line + '\n' for line in abc.split('\n')][:-1]
    abc_lines = [line for line in abc_lines if line.strip() != '']
    abc_lines = unidecode_abc_lines(abc_lines)
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])
    abc_lines = remove_bar_no_annotations(abc_lines)

    # delete \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        if r'\"' in line:
            abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # delete text annotations with quotes
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # check bar alignment
    with _shutup():
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
    if not bar_no_equal_flag:
        raise ValueError('Bar alignment error')

    # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line
    interleaved_abc: list[str] = rotate_abc(abc_lines)
    return interleaved_abc


def xml2abc(xml_path: str, raise_on_fail: bool = True) -> str:
    abc, err = run_xml2abc([xml_path])
    if raise_on_fail and not _MAYBE_SUCCESSFUL.search(err):
        raise ValueError(f"xml2abc likely failed with error: {err}")
    return abc


def run_xml2abc(input_files: list[str], unfold_repeats=False, midi_mode=0, credit_text_filter=0, line_ratio=0,
                max_chars_per_line=100, max_bars_per_line=0, output_directory='', volta_typesetting=0,
                no_line_breaks=False, page_format='', js_compatibility=False, translate_percussion=False,
                shift_note_heads=False, all_directions_to_first_voice=False, include_pedal_directions=True,
                translate_stem_directions=False, read_from_stdin=False, *, show_cmd: bool = False):
    """
    Executes the abc2xml.py script with given options and input files.

    Parameters:
    - unfold_repeats (bool): If True, unfolds simple repeats in the music.
    - midi_mode (int): Level of MIDI information to include (0: none, 1: minimal, 2: all).
    - credit_text_filter (int): Filters out credits in the music based on this integer value.
    - line_ratio (int): Sets the L:1/D ratio for line breaks.
    - max_chars_per_line (int): Maximum characters per line in the output.
    - max_bars_per_line (int): Maximum number of bars per line in the musical notation.
    - output_directory (str): Directory to store the output abc files.
    - volta_typesetting (int): Sets volta typesetting behavior.
    - no_line_breaks (bool): If True, no line breaks will be outputted.
    - page_format (str): Specifies the page format in detail.
    - js_compatibility (bool): If True, makes the output compatible with a JavaScript version.
    - translate_percussion (bool): If True, translates percussion and tab staff to ABC code.
    - shift_note_heads (bool): If True, shifts note heads leftwards in tab staff.
    - all_directions_to_first_voice (bool): If True, all directions apply only to the first voice.
    - include_pedal_directions (bool): If False, skips all pedal directions.
    - translate_stem_directions (bool): If True, translates stem directions.
    - read_from_stdin (bool): If True, reads input from standard input instead of files.
    - input_files (list of str): List of input files to process.

    Returns:
    - tuple: (stdout, stderr) from the subprocess execution.
    """
    xml2abc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "abc", "xml2abc.py"))
    if not os.path.exists(xml2abc_path):
        raise FileNotFoundError(f"xml2abc.py not found at {xml2abc_path}")

    cmd = [sys.executable, xml2abc_path]

    if not input_files:
        raise ValueError("input_files must be provided")

    if unfold_repeats:
        cmd.append('-u')
    if midi_mode != 0:
        cmd.extend(['-m', str(midi_mode)])
    if credit_text_filter != 0:
        cmd.extend(['-c', str(credit_text_filter)])
    if line_ratio != 0:
        cmd.extend(['-d', str(line_ratio)])
    if max_chars_per_line != 100:
        cmd.extend(['-n', str(max_chars_per_line)])
    if max_bars_per_line != 0:
        cmd.extend(['-b', str(max_bars_per_line)])
    if output_directory:
        cmd.extend(['-o', output_directory])
    if volta_typesetting != 0:
        cmd.extend(['-v', str(volta_typesetting)])
    if no_line_breaks:
        cmd.append('-x')
    if page_format:
        cmd.extend(['-p', page_format])
    if js_compatibility:
        cmd.append('-j')
    if translate_percussion:
        cmd.append('-t')
    if shift_note_heads:
        cmd.append('-s')
    if all_directions_to_first_voice:
        cmd.append('--v1')
    if not include_pedal_directions:
        cmd.append('--noped')
    if translate_stem_directions:
        cmd.append('--stems')
    if read_from_stdin:
        cmd.append('-i')

    cmd.extend(input_files)
    if show_cmd:
        print("Running command:", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr


def run_abc2xml(input_files: list[str], output_directory='', skip_tunes=0, max_tunes=1, page_format='',
                compression_mode='', show_whole_measure_rests=False, use_title_as_filename=False,
                line_break_at_eol=False, metadata_mapping='', force_string_allocation=False):
    """
    Converts ABC music notation files to XML format using the abc2xml.py script.

    Parameters:
    - output_directory (str): Directory to store the resulting XML files.
    - skip_tunes (int): Number of tunes to skip at the beginning.
    - max_tunes (int): Maximum number of tunes to process after skipping.
    - page_format (str): Page formatting string, expected to be 7 comma-separated values.
    - compression_mode (str): Whether to store output as compressed MXL ('add' or 'replace').
    - show_whole_measure_rests (bool): If True, show whole measure rests in merged staffs.
    - use_title_as_filename (bool): If True, use the tune title as the filename for output files.
    - line_break_at_eol (bool): If True, enforce line breaks at the end of lines.
    - metadata_mapping (str): Mapping of info fields to XML metadata.
    - force_string_allocation (bool): If True, force string/fret allocations for tab staves.
    - input_files (list of str): List of input ABC files to process.

    Returns:
    - stdout (str): Standard output from the script.
    - stderr (str): Standard error output, if any.
    """
    abc2xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "abc", "xml2abc.py"))
    if not os.path.exists(abc2xml_path):
        raise FileNotFoundError(f"xml2abc.py not found at {abc2xml_path}")

    cmd = [sys.executable, abc2xml_path]

    # Add options based on function arguments
    if output_directory:
        cmd.extend(['-o', output_directory])
    if (skip_tunes, max_tunes) != (0, 1):
        cmd.extend(['-m', str(skip_tunes), str(max_tunes)])
    if page_format:
        cmd.extend(['-p', page_format])
    if compression_mode:
        cmd.extend(['-z', compression_mode])
    if show_whole_measure_rests:
        cmd.append('-r')
    if use_title_as_filename:
        cmd.append('-t')
    if line_break_at_eol:
        cmd.append('-b')
    if metadata_mapping:
        cmd.extend(['--meta', metadata_mapping])
    if force_string_allocation:
        cmd.append('-f')

    # Add file arguments
    cmd.extend(input_files)

    # Execute the command using subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Return the output and errors
    return result.stdout, result.stderr
