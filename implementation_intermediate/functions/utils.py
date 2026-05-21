import json
import os
from contextlib import contextmanager
from json import JSONEncoder

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    # converts numpy array to a list so that it can be dumped into a .json file
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


# This function is used to suppress error messages during optimization
# which avoid cluttering the output with error messages
@contextmanager
def silent_errors(stdchannel, dest_filename):
    dest_file = None
    old_stdchannel = None
    try:
        old_stdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())
        yield
    finally:
        if old_stdchannel is not None:
            os.dup2(old_stdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


def load_best_parameters(folder, model=None, param_key = 'x', cost_key = 'f'):
    """Load the best parameters from a folder with json files. The best parameters are the ones with the lowest cost.
    The function assumes that the json files have the following structure:
    {
        "f": [cost of the parameters]
        "x": [list of parameters],
    }

    Args:
        folder (string): The folder to search for the results json files.
        model (sund.model, optional): Model object to load default values from. Defaults to None.
        param_key (str, optional): The key in the json file for parameter values. Defaults to 'x'.
        cost_key (str, optional): The key in the json file for the cost. Defaults to 'f'.

    Returns:
        np.ndarray: The best parameter values.
    """

    # Check if the folder exists and contains files
    results = []
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        files = os.listdir(folder)
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(f"{folder}/{file}", 'r') as f:
                        results.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"Could not load {folder}/{file}")
                except FileNotFoundError:
                    print(
                        f"The file '{folder}/{file}' does not exist. If it is a temporary file, it might have been removed and this error can be ignored.")
                except PermissionError:
                    print(
                        f"Permission denied for file '{folder}/{file}'. If it is a temporary file, it might have been removed and this error can be ignored.")

    # find the best results loaded from files
    if len(results) > 0:
        best_result = min(results, key=lambda x: x[cost_key])[param_key]
    else:
        print("No best parameters found.")
        if model is None:
            best_result = np.array([])
        else:
            print("Using default values from model.")
            best_result = np.array(model.parameter_values)

    return best_result


def reconstruct_parameter_vector(p_partial, theta0_constants_log = np.array([]), constant_indices = []):
    # Build the full vector by interleaving constant and optimized values
    p_full = []
    constant_set = set(constant_indices)
    constant_dict = dict(zip(constant_indices, theta0_constants_log))
    
    partial_idx = 0
    for i in range(len(p_partial) + len(constant_indices)):
        if i in constant_set:
            p_full.append(constant_dict[i])
        else:
            p_full.append(p_partial[partial_idx])
            partial_idx += 1
    
    return np.array(p_full)
