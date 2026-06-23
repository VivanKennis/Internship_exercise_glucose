import datetime
import os
import random
import sys
import time

import numpy as np

from common.cost_functions import f_cost
from common.parameter_estimation_functions import (
    ROUNDING_PRECISION, cleanup_temp_file, get_parameter_bounds,
    get_parameter_indices_to_optimize, run_directional_optimization,
    save_result, setup_simulations)
from common.utils import load_best_parameters, reconstruct_parameter_vector


######################################################
########################  PI  ########################
######################################################

# objective function for PI that allows for minimization (direction=1) and maximization (direction=-1)
def f_cost_PI_log(
    p_log: np.ndarray, sims: dict, data: dict, param_to_optimize_index: int, value_accepted: float, 
    limit: float, direction: int, simulate_steady_state: bool = False, 
    theta0_constants_log: np.ndarray = np.array([]), constant_indices: list = []) -> float:

    # if not all params are part of the optimization - reconstruct the full vector
    p_full = reconstruct_parameter_vector(
        p_log, theta0_constants_log, constant_indices)
    p = np.exp(p_full).round(ROUNDING_PRECISION)
    cost_agreement = f_cost(
        p, sims, data, simulate_steady_state=simulate_steady_state)  # find objective value

    # set PI objective value (if maximization we make the param value negative to effectively get the biggest possible "-1*positive" value)
    value = direction*p_log[param_to_optimize_index]

    # if the objective value exceeds the limit - add a punishment
    if cost_agreement > limit:
        value += (np.abs(value) + np.abs(value_accepted)) * \
            (1 + (cost_agreement - limit))

    return value


def find_least_tested_parameter(model_name:str, parameter_names: list[str], direction: int, optimize_indices: list) -> str:
    """
    Args:
        model_name (str): Name of the model.
        parameter_names (list): List of parameter names.
        direction (int): Define if the parameter should be minimized (1) or maximized (-1).
        optimize_indices (list): List of parameter indices that are to be optimized.

    Returns:
        string: The parameter name that has been the least tested.
    """
    files = os.listdir(f"./results_PI/{model_name}")
    
    # Find all parameters that have been tested, but ignore those that have already been minimized to 1e-20, or maximized to 1e20
    tested = []
    prefix = f"{model_name}-"
    for f in files:
        if not f.endswith(".json") or not f.startswith(prefix):
            continue
        param_name = f[len(prefix):].split("-")[0]
        if param_name not in parameter_names:
            continue
        if parameter_names.index(param_name) not in optimize_indices:
            continue
        if (not any(param_name in file and "e-20" in file for file in files) and direction == 1) or \
           (not any(param_name in file and "e20" in file for file in files) and direction == -1):
            tested.append(param_name)
        else:
            if direction == 1:
                print(f"Skipping {param_name} as it has already been found to be minimized (1e-20).")
            else:
                print(f"Skipping {param_name} as it has already been found to be maximized (1e20).")

    # find the least tested param
    min_count = min(map(tested.count, set(tested)))
    least_common_elements = [x for x in set(
        tested) if tested.count(x) == min_count]

    return random.choice(least_common_elements)


def optimize_PI(input_args: dict) -> None:
    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())

    # unpack input arguments
    data                     = input_args['data']
    fixed_parameters         = input_args['fixed_params']
    model_name               = input_args['model_name']
    simulate_steady_state    = input_args['simulate_steady_state']
    strict_bounds_parameters = input_args['strict_params']
    limit                    = input_args['chi2_limit']
    direction                = input_args['dir']

    # setup simulations
    sims = setup_simulations(model_name, data)
    sim = next(iter(sims.values()))

    # Load the overall best parameter set found so far with respect to the agreement to data
    theta_best = load_best_parameters(f"./results/{model_name}", cost_key='f')
    theta_best_log = np.log(theta_best)

    # Find what parameters to optimize
    optimize_indices, constant_indices = get_parameter_indices_to_optimize(
        sim.parameter_names, fixed_parameters)

    # Find which parameter to profile
    param_name = find_least_tested_parameter(model_name, sim.parameter_names, direction, optimize_indices)

    # temporary file to store the best parameters found so far
    temp_filename = f"./results_PI/{model_name}/{model_name}-{param_name}-temp-{os.getpid()}.json"

    # Setup values for optimizing the parameters
    param_to_optimize_index = sim.parameter_names.index(param_name)
    value_accepted = float(theta_best_log[param_to_optimize_index])

    # Load the best parameters with respect to minimization/maximization of the parameter name "param_name"
    # Fall back to global best parameters if no PI results exist yet for this parameter
    theta0 = load_best_parameters(
        f"./results_PI/{model_name}", key=f'-{param_name}-', cost_key='f', direction=direction)
    if len(theta0) == 0:
        theta0 = theta_best

    # divide start guess in a part to optimize and a part to remain constant
    theta0_log = np.log(np.array(theta0)[optimize_indices])
    theta0_constants_log = np.log(np.array(theta0)[constant_indices])

    # set up bounds
    bounds = get_parameter_bounds(
        sim.parameter_names, theta0_log, optimize_indices, strict_bounds_parameters)
    # Relax the bounds for the parameter to profile
    if direction == 1:
        bounds.lb[param_to_optimize_index] = np.log(1e-20)
    else:
        bounds.ub[param_to_optimize_index] = np.log(1e20)

    # make sure that the start guess is within bounds
    theta0_log = np.clip(theta0_log, bounds.lb, bounds.ub)

    # package arguments and find the starting PI value
    args = (
        sims, data, param_to_optimize_index, value_accepted, limit,
        direction, simulate_steady_state, theta0_constants_log, constant_indices)
    best_f_obj = f_cost_PI_log(theta0_log, *args)

    # run DE/DA optimization
    best_x = run_directional_optimization(
        objective_fn=f_cost_PI_log,
        args=args,
        theta0_log=theta0_log,
        bounds=bounds,
        input_args=input_args,
        theta0_constants_log=theta0_constants_log,
        constant_indices=constant_indices,
        best_initial_value=best_f_obj,
        temp_filename=temp_filename,
        da_label=f" - param: {param_name}",
    )

    p_full = reconstruct_parameter_vector(best_x, theta0_constants_log, constant_indices)

    # Save the final results
    optimized_param_value = float(np.exp(p_full[param_to_optimize_index]).round(ROUNDING_PRECISION))
    file_name = f"./results_PI/{model_name}/{model_name}-{param_name}-({optimized_param_value:e}) - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    save_result(optimized_param_value, p_full, file_name, sim.parameter_names)

    # Remove the temporary file created by the callback function
    cleanup_temp_file(file_path=temp_filename)
