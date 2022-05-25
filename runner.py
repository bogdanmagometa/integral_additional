"""
runner.py

This python module is a script for testing C++ program calculating the value of integral.
The module does two things:
1. Creates the configuration file.
2. Runs the executable specified number of times.

Run
$ python3 runner.py NUMBER_OF_RUNS
to execute the module. NUMBER_OF_RUNS should be replaced with the number of times you want 
your executable to be executed.

Specify the path to the executable in PATH_TO_EXECUTABLE global variable.

Specify the options passed to the executable by modifying the following global variables:
- PATH_TO_CONFIG_FILE
- CONFIG_FILE_DICT
"""

import os
import sys
import subprocess
from typing import Tuple, Union, Dict

VARIANT = 4
PATH_TO_CONFIG_FILE = './config_file_1cdXhU4KdaDdL.cfg.temp'

# ---- CHANGE IT TO YOUR PATH TO THE EXECUTABLE IF NEEDED ----
PATH_TO_EXECUTABLE = "./bin/integrate_task_4"

# ---- THIS DICT SPECIFIES OPTIONS IN THE TEMPORARY CONFIGURATION ----
# ---- FILE CREATED BY THE SCRIPT                                 ----
CONFIG_FILE_DICT = {
    'abs_err': 0.0000005,
    'rel_err': 0.00002,
    'x_start': -10,
    'x_end': 10,
    'y_start': -10,
    'y_end': 10,
    'init_steps_x': 100,
    'init_steps_y': 100,
    'max_iter': 10,
}

def main():
    num_runs = int(sys.argv[1])
    create_config_file(CONFIG_FILE_DICT, PATH_TO_CONFIG_FILE)

    min_possible_val = float("-inf")
    max_possible_val = float("inf")
    min_time = float("inf")
    for _ in range(num_runs):
        output = run_cpp_program(PATH_TO_EXECUTABLE, PATH_TO_CONFIG_FILE)
        if output == -1:
            delete_config_file(PATH_TO_CONFIG_FILE)
            return
        output_split = output.split()
        variant = int(output_split[0])
        calc_val = float(output_split[1])
        abs_err = float(output_split[2])
        rel_err = float(output_split[3])
        time_ms = int(output_split[4])
        if (variant != VARIANT):
            print("Incorrect variant.")
            delete_config_file(PATH_TO_CONFIG_FILE)
            return
        min_possible_val, max_possible_val = find_intersection(min_possible_val, max_possible_val, 
                                            calc_val - CONFIG_FILE_DICT["abs_err"], calc_val + CONFIG_FILE_DICT["abs_err"])
        if (min_possible_val > max_possible_val):
            print("At least one of the calculated values of integral is not "
                    f"withing absolute error")
            delete_config_file(PATH_TO_CONFIG_FILE)
            return
        if (abs_err > CONFIG_FILE_DICT['abs_err']):
            print("Absolute error calculated by the program is larger the required.")
        if (rel_err > CONFIG_FILE_DICT['rel_err']):
            print("Relative error calculated by the program is larger the required.")
        min_time = min(time_ms, min_time)
    print(min_time)
    delete_config_file(PATH_TO_CONFIG_FILE)


def create_config_file(config_file_dict: Dict[str, Union[int, float]], path_to_file: str) -> None:
    with open(path_to_file, 'w') as conf_f:
        for prop_name, prop_val in config_file_dict.items():
            print(prop_name, '=', prop_val, file = conf_f)


def delete_config_file(path_to_file: str) -> bool:
    if os.path.exists(path_to_file):
        os.remove(path_to_file)
        return True
    return False


def run_cpp_program(path_to_executable, path_to_config: str) -> Union[str, int]:
    """
    Execute C++ program and return the time of the execution got from
    stdout of the program.
    :return: The output of the program, -1 (int) if error occured
    """
    process = subprocess.Popen([path_to_executable, path_to_config], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    if stderr:
        print(stderr, end="")
        return -1
    
    return stdout


def find_intersection(start1: float, end1: float, start2: float, end2: float) -> Tuple[float, float]:
    """
    Find intersection between [start1, end1] and [start2, end2].
    :return: The 2-tuple of floats corresponding to the intersection, (1, 0) if no intersetion.
    """
    if (end1 < start2 or end2 < start1):
        return 1, 0
    return max(start1, start2), min(end1, end2)


if __name__ == "__main__":
    main()
