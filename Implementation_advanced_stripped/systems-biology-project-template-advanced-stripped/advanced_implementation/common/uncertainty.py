import datetime
import os
import time

import numpy as np

from common.cost_functions import f_cost
from advanced_implementation.common.parameter_estimation_functions import (
    ROUNDING_PRECISION, cleanup_temp_file, get_parameter_bounds,
    get_parameter_indices_to_optimize, run_directional_optimization,
    save_result, setup_simulations)
from common.utils import load_best_parameters, reconstruct_parameter_vector


######################################################
##############  Reverse PPL  #########################
######################################################

# objective function for reverse PPL that returns cost and sim value at a specific time point
def f_cost_reverse_PPL(
    p: np.ndarray, sims: dict, D: dict, exp_to_optimize: str, obs_to_optimize: str,
    t_to_optimize: float, simulate_steady_state: bool = False) -> tuple[float, float]:
    cost = f_cost(p, sims, D, simulate_steady_state)

    # If the cost is too large, return immediately - simulation has crashed
    if cost >= 1e20:
        return 1e20, cost

    try:
        # Proceed only if the simulation has not crashed
        sim = sims[exp_to_optimize]
        idx = sim.feature_names.index(obs_to_optimize)
        prediction_value = sim.feature_data[np.searchsorted(
            sim.time_vector, t_to_optimize), idx]

    except Exception as e:
        # Handle any errors and return a large value
        print(f"Error in {exp_to_optimize}: {e}")
        prediction_value = 1e20

    return prediction_value, cost


# objective function for reverse PPL that allows for minimization (direction=1) and maximization (direction=-1)
def f_cost_reverse_PPL_log(
    p_log: np.ndarray, sims: dict, data: dict, exp_to_optimize: str, obs_to_optimize: str,
    t_to_optimize: float, value_accepted: float, limit: float, direction: int,
    simulate_steady_state: bool = False, theta0_constants_log: np.ndarray = np.array([]), 
    constant_indices: list = []) -> float:

    p_full = reconstruct_parameter_vector(
        p_log, theta0_constants_log, constant_indices)

    p = np.exp(p_full).round(ROUNDING_PRECISION)
    prediction_value, cost_agreement = f_cost_reverse_PPL(
        p, sims, data, exp_to_optimize, obs_to_optimize, t_to_optimize, simulate_steady_state)

    # set objective value (if maximization we make the sim value negative to effectively get the biggest possible "-1*positive" value)
    value = direction * prediction_value

    # if the objective value exceeds the limit - add a punishment
    if cost_agreement > limit:
        value += (
            np.abs(value) + np.abs(value_accepted)) * \
            (1 + (cost_agreement - limit))

    return value


def optimize_reverse_PPL(input_args: dict) -> None:
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
    exp_to_optimize          = input_args['exp']
    obs_to_optimize          = input_args['obs']
    t_to_optimize            = input_args['time']

    # setup simulations
    sims = setup_simulations(model_name, data)
    sim = next(iter(sims.values()))

    # temporary file to store the best parameters found so far
    temp_filename = f"./results_reverse_PPL/{model_name}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/temp-{os.getpid()}.json"

    # Find what parameters to optimize
    optimize_indices, constant_indices = get_parameter_indices_to_optimize(
        sim.parameter_names, fixed_parameters)

    # Load the best parameters with respect to minimization/maximization of the prediction value
    # Fall back to global best parameters if no reverse PPL results exist yet
    theta0 = load_best_parameters(
        f"./results_reverse_PPL/{model_name}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}",
        key=f'{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}', cost_key='f', direction=direction)
    if len(theta0) == 0:
        theta0 = load_best_parameters(f"./results/{model_name}", cost_key='f')

    # divide start guess in a part to optimize and a part to remain constant
    theta0_log = np.log(np.array(theta0)[optimize_indices])
    theta0_constants_log = np.log(np.array(theta0)[constant_indices])

    # set up bounds
    bounds = get_parameter_bounds(
        sim.parameter_names, theta0_log, optimize_indices, strict_bounds_parameters)

    # make sure that the start guess is within bounds
    theta0_log = np.clip(theta0_log, bounds.lb, bounds.ub)

    # simulate the experiment once to be able to extract the sim value
    value_accepted_float, _ = f_cost_reverse_PPL(
        theta0, sims, data, exp_to_optimize, obs_to_optimize, t_to_optimize, simulate_steady_state)
    value_accepted = float(value_accepted_float)

    # package arguments and find the starting reverse PPL value
    args = (sims, data, exp_to_optimize, obs_to_optimize, t_to_optimize, value_accepted,
            limit, direction, simulate_steady_state, theta0_constants_log, constant_indices)
    best_reverse_PPL_value = f_cost_reverse_PPL_log(theta0_log, *args)

    # run DE/DA optimization
    best_x = run_directional_optimization(
        objective_fn=f_cost_reverse_PPL_log,
        args=args,
        theta0_log=theta0_log,
        bounds=bounds,
        input_args=input_args,
        theta0_constants_log=theta0_constants_log,
        constant_indices=constant_indices,
        best_initial_value=best_reverse_PPL_value,
        temp_filename=temp_filename,
        da_label=f" - {exp_to_optimize} - {obs_to_optimize} - time value: {t_to_optimize}",
        p_names=sim.parameter_names,
    )

    # reconstruct full parameter vector and find the optimized reverse PPL value
    p_full = reconstruct_parameter_vector(best_x, theta0_constants_log, constant_indices)
    p = np.exp(p_full).round(ROUNDING_PRECISION)
    optimized_reverse_PPL_value, _ = f_cost_reverse_PPL(
        p, sims, data, exp_to_optimize, obs_to_optimize, t_to_optimize, simulate_steady_state)

    # Save the final results
    file_name = (
        f"./results_reverse_PPL/{model_name}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/"
        f"{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize} opt_reverse_PPL({optimized_reverse_PPL_value:e}) - "
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    save_result(optimized_reverse_PPL_value, p_full, file_name, sim.parameter_names)

    # Remove the temporary file created by the callback function
    cleanup_temp_file(file_path=temp_filename)