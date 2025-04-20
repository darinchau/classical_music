# Extracts features using jsymbolic
# Get jsymbolic from https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/jSymbolic_2_2_user.zip/download
# Extract the zip file and place the extracted directory inside resources/

import os
import subprocess
import tempfile
import csv
from ..reps import SymbolicMusic


def _require_java():
    """
    Check if Java is installed and available in the system PATH.
    """
    try:
        subprocess.run(['java', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise RuntimeError("Java is not installed or not available in the system PATH. Please install Java and try again.")


def _require_jsymbolic():
    """
    Check if jsymbolic is installed and available in the system PATH.
    """
    jsymbolic_path = os.path.join(os.path.dirname(__file__), 'resources', 'jSymbolic_2_2_user', 'jSymbolic.jar')
    if not os.path.exists(jsymbolic_path):
        raise RuntimeError("jsymbolic is not installed. Please download it from https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/jSymbolic_2_2_user.zip/download and place it in the resources directory.")
    return jsymbolic_path


def _run_jsymbolic(input_file: str):
    _require_java()
    jsymbolic_path = _require_jsymbolic()

    with tempfile.TemporaryDirectory() as temp_dir:
        # output = os.path.join(os.path.dirname(input_file), 'test_features.xml')
        # output_def = os.path.join(os.path.dirname(input_file), 'test_definitions.xml')
        # output_csv = os.path.join(os.path.dirname(input_file), 'test_features.csv')
        output = os.path.join(temp_dir, 'test_features.xml')
        output_def = os.path.join(temp_dir, 'test_definitions.xml')
        output_csv = os.path.join(temp_dir, 'test_features.csv')
        cmd = [
            "java",
            "-jar",
            jsymbolic_path,
            "-csv",
            os.path.abspath(input_file),
            os.path.abspath(output),
            os.path.abspath(output_def),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"jsymbolic failed with error: {e.stderr.decode()}") from e

        with open(output_csv, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            single_row_dict = next(reader, None)
            if single_row_dict is None:
                raise RuntimeError("No data found in the CSV output file.")
    return single_row_dict
