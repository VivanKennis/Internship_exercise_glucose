import datetime
import json
import os
import sys
import time

import numpy as np
import sund
from scipy.optimize import Bounds, differential_evolution, dual_annealing

from functions.cost_functions import f_cost_log_with_fixed_parameters
from functions.utils import (NumpyArrayEncoder, load_best_parameters,
                          reconstruct_parameter_vector, silent_errors)


def load_and_install_model(model_name, install_model=True):
    if install_model:
        sund.install_model(f"./models/{model_name}.txt")
    return sund.load_model(model_name)


def setup_simulations(model_name, data):
    from functions.simulation import create_sims_from_data
    model = load_and_install_model(model_name, install_model=False) 
    sims = create_sims_from_data(model, data)
    
    return sims


def get_parameter_indices_to_optimize(parameter_names, fixed_parameters = []):
    constant_indices, optimize_indices = [], []

    for idx, pName in enumerate(parameter_names):
        if pName in fixed_parameters:
            constant_indices.append(idx)
        else:
            optimize_indices.append(idx)

    return optimize_indices, constant_indices


def get_parameter_bounds(parameter_names, theta, optimize_indices):
    # Set up main parameter bounds
    lb = [np.log(1.0e-5)]*(len(theta))
    ub = [np.log(1.0e5)] * (len(theta))

    # Parameter-specific bounds
    param_bounds = {}
    param_bounds["ns"] = (0.1, 4) 
    # param_bounds["k2"] = (np.log(1e-2), np.log(1e2)) # Example of setting specific bounds for parameter "k2"

    # Apply parameter-specific bounds
    for param, (lower, upper) in param_bounds.items():
        if param in parameter_names and parameter_names.index(param) in optimize_indices:
            idx = optimize_indices.index(parameter_names.index(param))
            if lower is not None:
                lb[idx] = lower
            if upper is not None:
                ub[idx] = upper

    # Convert to numpy arrays and return bounds object
    bounds = Bounds(np.asarray(lb), np.asarray(ub))  # type: ignore[arg-type]
    return bounds


def save_result(cost, p_full, filename):
    try:
        with open(filename, 'w') as file:
            out = {"f": cost, "x": np.exp(p_full)}
            json.dump(out, file, cls=NumpyArrayEncoder)
    except PermissionError:
        print(
            f"Permission denied for file '{filename}'. Not saving current solution.")


def cleanup_temp_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except FileNotFoundError:
        print(
            f"Could not remove temporary file '{file_path}'. It might have been removed already.")


def optimize(input_args):
    #%% Define callback functions
    def callback_differential_evolution(x, convergence):
        cost = f_cost_log_with_fixed_parameters(
            x, sims, data, simulate_steady_state, theta0_constants_log, constant_indices)
        
        p_full = reconstruct_parameter_vector(
            x, theta0_constants_log, constant_indices)
        
        # save temporary result
        save_result(cost, p_full, temp_filename)

    def callback_dual_annealing(x, cost, context):
        nonlocal n_iter  # To access variable outside the function
        if print_iterations:
            print(
                f"dual_annealing step {n_iter}: f(x) = {cost} - pid= {os.getpid()}")
        n_iter += 1
        
        p_full = reconstruct_parameter_vector(
            x, theta0_constants_log, constant_indices)
        
        # save temporary result
        save_result(cost, p_full, temp_filename)

    #%% prepare for optimization
    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())

    # unpack input arguments
    data                     = input_args['data']
    fixed_parameters         = input_args['fixed_params']
    model_name               = input_args['model_name']
    simulate_steady_state    = input_args['simulate_steady_state']
    print_iterations         = input_args['print_iter']
    
    # set up simulations
    sims = setup_simulations(model_name, data)
    sim = next(iter(sims.values()))  # get one simulation to extract model information
    
    # temporary file to store the best parameters found so far
    temp_filename = f"./results/{model_name}/{model_name}-temp-{os.getpid()}.json"
    
    # Use model defaults when no previous result file exists.
    theta0 = load_best_parameters(f"./results/{model_name}", model=sim)
    
    optimize_indices, constant_indices = get_parameter_indices_to_optimize(
        sim.parameter_names, fixed_parameters)

    # divide start guess in a part to optimize and a part to remain constant
    theta0_log = np.log(np.array(theta0)[optimize_indices])
    theta0_constants_log = np.log(np.array(theta0)[constant_indices])
    
    # set up param bounds
    bounds = get_parameter_bounds(
        sim.parameter_names, theta0_log, optimize_indices)

    # if perturbation of start guess should occur - allow a change from theta0
    if np.random.random_sample() < input_args['perturbation_ratio']:
        theta0_log = theta0_log * \
            np.random.uniform(0.95, 1.05, len(theta0_log))

    # make sure the params is within bounds
    theta0_log = np.clip(theta0_log, bounds.lb, bounds.ub)

    # package input to objective function
    args = (sims, data, simulate_steady_state,
        theta0_constants_log, constant_indices)

    #%% Run the optimization
    # randomly choose optimization method based on the provided ratio
    do_differential_evolution = bool(
        np.random.random_sample() <= input_args['de_ratio'])

    if do_differential_evolution:
        print(f'Starting differential_evolution for pid {os.getpid()}')

        with silent_errors(sys.stderr, os.devnull):
            res = differential_evolution(
                func=f_cost_log_with_fixed_parameters, args=args, bounds=bounds,
                x0=theta0_log, disp=print_iterations, maxiter=input_args['max_iter'], 
                callback=callback_differential_evolution)  # type: ignore[arg-type]
    else:
        print(f'Starting dual_annealing for pid {os.getpid()}')
        n_iter = 1  # used for printing the current iteration in the dual_annealing callback

        with silent_errors(sys.stderr, os.devnull):
            res = dual_annealing(
                func=f_cost_log_with_fixed_parameters, args=args, bounds=bounds,
                x0=theta0_log, maxiter=input_args['max_iter'], callback=callback_dual_annealing)
        print(f'Done with dual_annealing for pid {os.getpid()}')

    #%% Save the final results
    file_name = f"./results/{model_name}/{model_name} ({res['fun']}) - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p_full = reconstruct_parameter_vector(
        res['x'], theta0_constants_log, constant_indices)
    save_result(res['fun'], p_full, file_name)

    # Remove the temporary file created by callback functions
    cleanup_temp_file(file_path=temp_filename)
