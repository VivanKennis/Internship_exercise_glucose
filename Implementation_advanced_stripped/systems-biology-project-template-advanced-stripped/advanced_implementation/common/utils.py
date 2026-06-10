import json
import os
from contextlib import contextmanager
from json import JSONEncoder
from typing import Any

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    # converts numpy array to a list so that it can be dumped into a .json file
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


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


def _order_by_name(best_result, old_mapping_dict, new_parameter_names) -> np.ndarray:
    """
    Reconstruct a parameter vector for the current order using name-based lookup.
    If values are missing, they are defaulted to 1.0, and a message is printed for each new parameter.
    """

    result = []
    new_params: list[tuple[str, str]] = []
    for name in new_parameter_names:
        if name in old_mapping_dict:
            result.append(best_result[old_mapping_dict[name]])
        else:
            result.append(1.0)
            print(f"Parameter '{name}' not found in best result. Defaulting to 1.0.")

    return np.array(result)


def load_best_parameters(
    folder: str, key: str | None = None, param_key: str = 'x', cost_key: str = 'cost', 
    print_names: bool = False, model: Any | None = None, direction: int = 1) -> np.ndarray:
    """Load the best parameters from a folder with json files. The best parameters are the ones with the lowest cost.
    The function assumes that the json files have the following structure:
    {
        "cost": [cost of the parameters]
        "x": [list of parameters],
    }

    Args:
        folder (string): The folder to search for the results json files.
        key (str, optional): A key that the files must contain to be considered. Defaults to None.
        param_key (str, optional): The key in the json file for parameter values. Defaults to 'x'.
        cost_key (str, optional): The key in the json file for the cost. Defaults to 'cost'.
        print_names (bool, optional): Print the parameter names if True.
        model (sund.model, optional): Model object to load default values from.
        direction (int, optional): Defines if the lowest or greatest solution should be loaded. 

    Returns:
        list: The best parameter values.
    """

    if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
        print("No best parameters found.")
        if model is None:
            return np.array([])
        else:
            print(f"Using default values from model {getattr(model, 'name', type(model).__name__)}")
            return np.array(model.parameter_values)

    files = os.listdir(folder)

    if key is not None:
        files = [f for f in files if key in f]

    results = []
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
        if direction == 1:
            best_result = min(results, key=lambda x: x[cost_key])
        else:
            best_result = max(results, key=lambda x: x[cost_key])

        if print_names:
            print(best_result)

        if "param_ids_dict" in best_result and model is not None:
            return _order_by_name(best_result[param_key], best_result["param_ids_dict"], model.parameter_names)
        
        return np.array(best_result[param_key])
    else:
        print("No best parameters found.")
        if model is None:
            return np.array([])
        else:
            print(f"Using default values from model {getattr(model, 'name', type(model).__name__)}")
            return np.array(model.parameter_values)


def reconstruct_parameter_vector(p_partial: np.ndarray, theta0_constants_log: np.ndarray = np.array([]), constant_indices: list[int] = []) -> np.ndarray:
    # Build the full vector by interleaving constant and optimized values
    p_full: list[float] = []
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
