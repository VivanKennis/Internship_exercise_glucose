import copy
import json
import os
from pathlib import Path

import numpy as np

from common.cost_functions import f_cost
from common.utils import (NumpyArrayEncoder, load_best_parameters)

ROUNDING_PRECISION = 15 # offset pythons default rounding error of 2.2e-16 when exp()

# objective function for reverse PPL that returns cost and sim value at a specific time point


def f_cost_reverse_PPL(p: np.ndarray, sims: dict, D: dict, model, exp_to_optimize: str, obs_to_optimize: str, t_to_optimize: float, simulate_steady_state: bool = False) -> tuple[float, float]:
    cost = f_cost(p, sims, D, model, simulate_steady_state)
    
    # If the cost is too large, return immediately - simulation has crashed
    if cost >= 1e20:
        return 1e20, cost
    
    try:
        sim = sims[exp_to_optimize]
        idx = sim.feature_names.index(obs_to_optimize)
        # find the nearest simulated time index (searchsorted can return len(tvec) if t is past the end)
        t_idx = int(np.argmin(np.abs(np.array(sim.time_vector) - t_to_optimize)))
        prediction_value = sim.feature_data[t_idx, idx]

    except Exception as e:
        print(f"Error in {exp_to_optimize}: {e}")
        prediction_value = 1e20

    return prediction_value, cost


def setup_reverse_PPL(model, sims: dict, data: dict, chi2_limit: float, validation_experiments: set[str], simulate_steady_state: bool = False) -> tuple[dict, dict]:
    # Initialize folder
    # make sure the reverse PPL results folder exists
    os.makedirs(f"./results_reverse_PPL/{model.name}", exist_ok=True)

    # do not calculate cost for validation experiments
    ALL_DATA_reverse_PPL = copy.deepcopy(data)
    for k_exp, d in ALL_DATA_reverse_PPL.items():
        if k_exp in validation_experiments:
            for k_obs in d["Observables"]:
                d["Observables"][k_obs]["SEM"] = [
                    np.inf for _ in range(0, len(d["Observables"][k_obs]["SEM"]))]

    # find number of combinations to iterate over
    reverse_PPL_combinations = {}
    iter = 0
    for k_exp, d in data.items():
        t_first_activity = min(
            (t for _, inp in d["input"].items() for t in inp["t"] if t != float('-inf')))
        for k_obs, obs in d["Observables"].items():
            for t in obs["Time"]:
                if t > t_first_activity:
                    reverse_PPL_combinations[iter] = {
                        "k_exp": k_exp, "k_obs": k_obs, "Time": t}
                    iter += 1

    # load best solution
    # The best parameters found so far with respect to the agreement to data
    theta_best = load_best_parameters(
        f"./results/{model.name}", cost_key='f', model=model)

    for k_exp, d in data.items():
        for k_obs, obs in d["Observables"].items():
            for t in obs["Time"]:
                # Initialize subfolder
                # make sure the reverse PPL results folder exists
                if not os.path.exists(f"./results_reverse_PPL/{model.name}/{k_exp}_{k_obs}_{t}"):
                    os.makedirs(
                        f"./results_reverse_PPL/{model.name}/{k_exp}_{k_obs}_{t}", exist_ok=True)

                    # find reverse PPL cost and save the objective value if the cost is lower than chi2-limit
                    reverse_PPL_obj, cost = f_cost_reverse_PPL(
                        theta_best, sims, data, model, k_exp, k_obs, t, simulate_steady_state)
                    if cost > chi2_limit:
                        print(
                            f"cost > limit: not saving value for: {k_exp}-{k_obs}-time_value-{t}")
                    else:
                        try:
                            with open(f"./results_reverse_PPL/{model.name}/{k_exp}_{k_obs}_{t}/{k_exp}_{k_obs}_{t} initial({reverse_PPL_obj:e}).json", 'w') as file:
                                out = {"f": reverse_PPL_obj, "x": theta_best}
                                json.dump(out, file, cls=NumpyArrayEncoder)
                        except PermissionError:
                            print(
                                f"Permission denied for file './results_reverse_PPL/{model.name}/{k_exp}_{k_obs}_{t}/{k_exp}_{k_obs}_{t}.json'. Not saving current solution.")

    return ALL_DATA_reverse_PPL, reverse_PPL_combinations


def collect_reverse_PPL(model, sims: dict, data: dict) -> None:
    # create folder to store UC results
    os.makedirs(f"./results/{model.name}/UC", exist_ok=True)

    # create a dictionary to store results
    t_step = 1
    UC_data = {}
    for k_exp, d in data.items():
        UC_data[k_exp] = {}
        for k_obs, _ in d["Observables"].items():
            UC_data[k_exp][k_obs] = {}
            t_vec = np.arange(0, d["all_times"][-1] + t_step, t_step)
            UC_data[k_exp][k_obs]["Time"] = t_vec
            UC_data[k_exp][k_obs]["Max"] = [float('-inf')] * len(t_vec)
            UC_data[k_exp][k_obs]["Min"] = [float('inf')] * len(t_vec)

    # loop over the reverse PPL results folder and add the results to the UC structure
    root_directory = Path(f"./results_reverse_PPL/{model.name}")

    files = []
    for item in root_directory.rglob("*"):  # Use rglob('*') for everything
        if item.is_file():
            files.append(item._str.replace('\\', '/'))

    # iterate over every file
    for file in files:
        if file.endswith(".json"):
            with open(f"{file}", 'r') as f:
                res = json.load(f)

                for k_exp, d in data.items():
                    
                    # High resolution time vector for smooth simulation curves
                    t_start = 0
                    t_end = d["all_times"][-1]
                    num_steps = int((t_end - t_start) / 0.1) + 1
                    t_high_res = np.linspace(t_start, t_end, num_steps)

                    sim = sims[k_exp]
                    sim.reset_states()
                    sim.simulate(
                        time_vector=t_high_res, parameter_values=res['x'], state_values=model.state_values)

                    for k_obs, _ in d["Observables"].items():
                        idx = sim.feature_names.index(k_obs)
                        y_sim = sim.feature_data[:, idx]

                        # do not save UC if simulation stayed at basal - if stimulation haven't trigger params can be any value
                        UC_data[k_exp][k_obs]["Max"] = np.maximum(
                            UC_data[k_exp][k_obs]["Max"], y_sim)
                        UC_data[k_exp][k_obs]["Min"] = np.minimum(
                            UC_data[k_exp][k_obs]["Min"], y_sim)

    # save the uncertainty
    with open(f"results/{model.name}/UC/UC_reverse_PPL_{model.name}.json", 'w') as f:
        json.dump(UC_data, f, cls=NumpyArrayEncoder)
