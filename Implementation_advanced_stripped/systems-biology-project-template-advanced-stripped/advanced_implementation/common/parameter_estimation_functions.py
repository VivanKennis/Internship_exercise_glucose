import copy
import datetime
import json
import os
import time

import numpy as np
from scipy.optimize import Bounds
from scipy.stats import qmc

from common.ess import _run_ess
from common.cost_functions import (
    f_cost, f_cost_log_with_fixed_parameters, f_residuals_log_with_fixed_parameters)
from common.utils import (
    NumpyArrayEncoder, load_best_parameters, reconstruct_parameter_vector)

ROUNDING_PRECISION = 15 # offset pythons default rounding error of 2.2e-16 when exp()


def setup_simulations(model_name: str, data: dict) -> dict:
    from methods.setup import load_and_install_model
    from common.simulation import create_sims_from_data
    model = load_and_install_model(model_name, install_model=False) 
    sims = create_sims_from_data(model, data)
    
    return sims


def get_parameter_indices_to_optimize(parameter_names: list[str], fixed_parameters: list[str] = []) -> tuple[list[int], list[int]]:
    constant_indices, optimize_indices = [], []

    for idx, pName in enumerate(parameter_names):
        if pName in fixed_parameters:
            constant_indices.append(idx)
        else:
            optimize_indices.append(idx)

    return optimize_indices, constant_indices


def get_parameter_bounds(parameter_names, theta, optimize_indices, strict_bounds_parameters=[]):

    # Setup main parameter bounds
    lb = [np.log(1.0e-5)]*(len(theta))
    ub = [np.log(1.0e5)] * (len(theta))

    # Parameter-specific bounds
    param_bounds = {}

    #param_bounds["vmax_o2"] = (np.log(1.0e-1), np.log(1e1))
    #param_bounds["vmax_lac"] = (np.log(1.0e-2), np.log(1e3)) 
    #param_bounds["n_lac"] = (np.log(1.0e0), np.log(1e1)) 
    #param_bounds["km_lac"] = (np.log(5.0e-3), np.log(1e1))
    #param_bounds["k_restore"] = (np.log(1.0e-3), np.log(1e3))  
    #param_bounds["elim_lactate"] = (np.log(1.0e-3), np.log(1e2))
    #param_bounds["k_O2"] = (np.log(1e-3), np.log(1e-1))    

    param_bounds["k_stress"] = (np.log(1e-3), np.log(1e-1))  
    param_bounds["elim_stress"] = (np.log(1.0e-4), np.log(1e1))  
    param_bounds["lac_stress"] = (np.log(1.0e-2), np.log(1e2))  

    param_bounds["k_push"] = (np.log(1.0e-6), np.log(1e2))  
    param_bounds["k_push2"] = (np.log(1.0e-6), np.log(1e2))  
    param_bounds["k_rec_drive"] = (np.log(1.0e-5), np.log(1e6)) 

    param_bounds["Vmax_stres_n"] = (np.log(1.0e-1), np.log(1e2))  
    param_bounds["km_stres_n"] = (np.log(1.0e-2), np.log(1e3))  
    param_bounds["n_stres_n"] = (np.log(1.0e0), np.log(4e0)) 

    param_bounds["spill"] = (np.log(1.0e-3), np.log(1e0))  
    param_bounds["elim_NOR_neuronal"] = (np.log(1.0e-2), np.log(1e1))  

    param_bounds["Vmax_stres_a"] = (np.log(1.0e-1), np.log(1e2)) 
    param_bounds["km_stres_a"] = (np.log(1.0e-2), np.log(1e3))  
    param_bounds["n_stres_a"] = (np.log(1.0e0), np.log(4e0))  
    
    param_bounds["Vmax_ex_n"] = (np.log(1.0e-1), np.log(1e2))  
    param_bounds["km_ex_n"] = (np.log(1.0e-2), np.log(1e3))  
    param_bounds["n_ex_n"] = (np.log(1.0e0), np.log(4e0)) 

    param_bounds["Vmax_ex_a"] = (np.log(1.0e-1), np.log(1e3))  
    param_bounds["km_ex_a"] = (np.log(1.0e-3), np.log(1e3))  
    param_bounds["n_ex_a"] = (np.log(1.0e0), np.log(4e0))  
    param_bounds["scale"] = (np.log(1.0e-5), np.log(1e2))  
    param_bounds["conv"] = (np.log(1.0e-5), np.log(1e0)) 

    param_bounds["release_epi"] = (np.log(1.0e-4), np.log(1e0))  
    param_bounds["elim_nor_plasma"] = (np.log(1.0e-2), np.log(1e1))  
    param_bounds["elim_epi_plasma"] = (np.log(1.0e-2), np.log(1e1))
   
    # Apply parameter-specific bounds
    for param, (lower, upper) in param_bounds.items():
        if param in parameter_names and parameter_names.index(param) in optimize_indices:
            idx = optimize_indices.index(parameter_names.index(param))
            if lower is not None:
                lb[idx] = lower
            if upper is not None:
                ub[idx] = upper
    
    # if parameters with strict bounds are provided - reduce the allowed parameter space
    lb_original = copy.deepcopy(lb)
    ub_original = copy.deepcopy(ub)

    for param_name in strict_bounds_parameters:
        if param_name in parameter_names:
            idx = parameter_names.index(param_name)
            lb[idx] = np.log(np.exp(theta[idx]) * 0.95)
            ub[idx] = np.log(np.exp(theta[idx]) * 1.05)

    # Ensure that none of the stricter/fixed bounds are outside the original bounds
    ub = np.minimum(ub, ub_original)
    lb = np.maximum(lb, lb_original)

    # Convert to numpy arrays and return bounds object
    bounds = Bounds(np.asarray(lb), np.asarray(ub))  # type: ignore[arg-type]
    return bounds


def create_init_pop(theta0_log: np.ndarray, bounds: Bounds, pop_size: int) -> np.ndarray:
    init = []
    samples_theta0 = int(pop_size*0.1)
    samples_qmc = int(pop_size*0.9)

    # add start guesses close to theta0
    while len(init) < samples_theta0:
        theta_guess = np.log(
            np.exp(theta0_log) * np.random.uniform(0.95, 1.05, len(theta0_log)))
        # make sure guess is within bounds
        init.append(np.clip(theta_guess, bounds.lb, bounds.ub))

    # add start guesses sampled within the bounds
    sampler = qmc.LatinHypercube(d=len(theta0_log))
    sample = sampler.random(n=samples_qmc)
    sample_scaled = qmc.scale(sample, bounds.lb, bounds.ub)

    [init.append(sample) for sample in sample_scaled]

    return np.array(init)


def bounds_to_tuples(lb: np.ndarray, ub: np.ndarray, len_bounds: int) -> list[tuple[float | None, float | None]]:
    lb_expanded = np.broadcast_to(lb, len_bounds)
    ub_expanded = np.broadcast_to(ub, len_bounds)

    lb_list = [float(x) if x > -np.inf else None for x in lb_expanded]
    ub_list = [float(x) if x < np.inf else None for x in ub_expanded]

    return list(zip(lb_list, ub_list))


def save_result(cost: float, p_full: np.ndarray, filename: str, parameter_names: list[str]) -> None:
    try:
        p = np.exp(p_full).round(ROUNDING_PRECISION)
        param_ids_dict = {}
        [param_ids_dict.update({name: idx}) for idx, name in enumerate(parameter_names)]
        with open(filename, 'w') as file:
            out = {
                "f": cost,
                "x": p,
                "param_ids_dict": param_ids_dict,
            }
            json.dump(out, file, cls=NumpyArrayEncoder)
    except PermissionError:
        print(
            f"Permission denied for file '{filename}'. Not saving current solution.")


def cleanup_temp_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except FileNotFoundError:
        print(
            f"Could not remove temporary file '{file_path}'. It might have been removed already.")


######################################################
########  Directional (PI / reverse PPL)  ############
######################################################

def run_directional_optimization(
    objective_fn,
    args: tuple,
    theta0_log: np.ndarray,
    bounds: Bounds,
    input_args: dict,
    theta0_constants_log: np.ndarray,
    constant_indices: list,
    best_initial_value: float,
    temp_filename: str,
    da_label: str = "",
    p_names: list[str] = [],
) -> np.ndarray:
    """
    Run DE or DA optimisation for directional (PI / reverse PPL) problems.

    Randomly selects between differential_evolution and dual_annealing based on
    ``input_args['de_ratio']``.  A checkpoint is written to *temp_filename*
    whenever a new best objective value is found.

    Returns the best parameter sub-vector found (optimized subspace, log-space).
    """
    import sys
    from scipy.optimize import differential_evolution, dual_annealing
    from common.utils import silent_errors

    print_iterations = input_args['print_iter']
    best_value = best_initial_value

    def callback(x: np.ndarray, convergence) -> None:
        nonlocal best_value
        value = objective_fn(x, *args)
        if value < best_value:
            best_value = value
            p_full = reconstruct_parameter_vector(x, theta0_constants_log, constant_indices)
            save_result(value, p_full, temp_filename, p_names)

    n_iter_da = 1

    def callback_dual_annealing(x: np.ndarray, value: float, context) -> None:
        nonlocal n_iter_da, best_value
        if print_iterations:
            print(f"dual_annealing step {n_iter_da}: f(x) = {value} - pid= {os.getpid()}{da_label}")
        n_iter_da += 1
        if value < best_value:
            best_value = value
            p_full = reconstruct_parameter_vector(x, theta0_constants_log, constant_indices)
            save_result(value, p_full, temp_filename, p_names)

    do_de = bool(np.random.random_sample() <= input_args['de_ratio'])

    if do_de:
        print(f'Starting differential_evolution for pid {os.getpid()}')
        init = create_init_pop(theta0_log, bounds, input_args['pop_size'])
        with silent_errors(sys.stderr, os.devnull):
            res = differential_evolution(
                func=objective_fn, args=args, bounds=bounds, x0=theta0_log,
                disp=print_iterations, maxiter=input_args["max_iter"],
                callback=callback, init=init)  # type: ignore[arg-type]
    else:
        print(f'Starting dual_annealing for pid {os.getpid()}')
        da_bounds = bounds_to_tuples(bounds.lb, bounds.ub, len(bounds.lb))
        with silent_errors(sys.stderr, os.devnull):
            res = dual_annealing(
                func=objective_fn, args=args, bounds=da_bounds, x0=theta0_log,
                maxiter=input_args["max_iter"], callback=callback_dual_annealing)

    return res['x']


######################################################
##########    Parameter Estimation(Opt)    ###########
######################################################


def optimize(input_args: dict) -> None:
    """
    Run a single ESS optimisation. Common initialisation then delegates to
    `_run_ess` for the algorithm itself.
    """
    import functools
    from math import ceil, sqrt

    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())

    # unpack input arguments
    data                     = input_args['data']
    fixed_parameters         = input_args['fixed_params']
    model_name               = input_args['model_name']
    simulate_steady_state    = input_args['simulate_steady_state']
    strict_bounds_parameters = input_args['strict_params']
    max_eval                 = input_args.get('ess_max_eval', 100_000)
    dim_refset               = input_args.get('ess_dim_refset', 10)
    local_n1                 = input_args.get('ess_local_n1', 1)
    local_n2                 = input_args.get('ess_local_n2', 2)
    balance                  = input_args.get('ess_balance', 0.5)
    max_walltime_s           = input_args.get('ess_max_walltime_s', None)
    use_finish_polish        = input_args.get('use_finish_polish', True)
    finish_polish_maxfev     = input_args.get('finish_polish_maxfev', 5000)
    vtr                      = input_args.get('ess_vtr', None)
    n_stuck_global_limit     = input_args.get('ess_n_stuck_global', 0)
    theta0_include_prob      = input_args.get('ess_theta0_include_prob', 1.0)
    local_solver             = input_args.get('ess_local_solver', 'nelder-mead')
    n_diverse_factor         = input_args.get('ess_n_diverse_factor', 10)
    stagnation_perturb_sigma = input_args.get('ess_stagnation_perturb_sigma', 0.1)
    n_ls_children            = input_args.get('ess_n_ls_children', 0)
    kick_limit               = input_args.get('ess_kick_limit', 3)
    save_trace               = input_args.get('ess_save_trace', False)
    disk_inject_enabled      = input_args.get('ess_disk_inject_enabled', True)
    inject_warmup_iters      = input_args.get('ess_inject_warmup_iters', 50)
    print_iterations         = input_args['print_iter']

    _results_dir  = f"./results/{model_name}"
    temp_filename = f"{_results_dir}/{model_name}-temp-{os.getpid()}.json"
    os.makedirs(_results_dir, exist_ok=True)

    # setup simulations
    sims = setup_simulations(model_name, data)
    sim  = next(iter(sims.values()))

    theta0 = load_best_parameters(_results_dir, cost_key='f', model=sim)

    optimize_indices, constant_indices = get_parameter_indices_to_optimize(
        sim.parameter_names, fixed_parameters)

    theta0_array         = np.array(theta0)
    theta0_log           = np.log(theta0_array[optimize_indices])
    theta0_constants_log = np.log(theta0_array[constant_indices])

    bounds = get_parameter_bounds(
        sim.parameter_names, theta0_log, optimize_indices, strict_bounds_parameters)
    lb = np.array(bounds.lb)
    ub = np.array(bounds.ub)
    n_params = len(theta0_log)

    # Auto-scale dim_refset: ensure it is large enough to represent the space.
    # Formula from Egea & Banga (2009): max(5, ceil((1 + sqrt(4n)) / 2)).
    dim_refset = max(dim_refset, max(5, ceil((1 + sqrt(4 * n_params)) / 2)))

    # Auto-scale evaluation budget: target ~1000·n total evaluations.
    max_eval = max(max_eval, 1000 * n_params)

    # Scale finish-polish budget with problem size.
    finish_polish_maxfev = max(finish_polish_maxfev, 500 * n_params)

    # Bind fixed-parameter context so the callable only sees the optimized sub-vector.
    objective_fn = functools.partial(
        f_cost_log_with_fixed_parameters,
        sims=sims,
        data=data,
        simulate_steady_state=simulate_steady_state,
        theta0_constant=theta0_constants_log,
        constant_indices=constant_indices,
    )
    residuals_fn = functools.partial(
        f_residuals_log_with_fixed_parameters,
        sims=sims,
        data=data,
        simulate_steady_state=simulate_steady_state,
        theta0_constant=theta0_constants_log,
        constant_indices=constant_indices,
    )

    # Test evaluation at theta0 — abort early if the model is broken.
    test_cost = objective_fn(theta0_log)
    if test_cost >= 1e20:
        print(
            f"  WARNING: test evaluation at theta0 returned {test_cost:.3e}. "
            f"Check parameter assembly and SUND model. Aborting.")
        return
    print(f"  Test evaluation at theta0: f = {test_cost:.6g}")

    # Convergence trace (list of (n_evals, elapsed_s, cost) tuples).
    _trace: list | None = [] if save_trace else None

    # Checkpoint callback: called by _run_ess on every new global best.
    def _on_new_best(cost: float, x_opt: np.ndarray) -> None:
        try:
            p_full = reconstruct_parameter_vector(x_opt, theta0_constants_log, constant_indices)
            save_result(cost, p_full, temp_filename, sim.parameter_names)
        except Exception as e:
            print(f"Error saving ESS checkpoint: {e}")

    # Disk-inject callback: scans the results folder and returns the best
    # improvement found by other parallel workers, or None.
    def _disk_inject():
        best_cost  = float('inf')
        best_x_log = None
        for fname in os.listdir(_results_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(_results_dir, fname)) as fh:
                    result = json.load(fh)
                cost = result.get('f', float('inf'))
                if not isinstance(cost, (int, float)) or cost >= best_cost:
                    continue
                full_x_log = np.log(np.maximum(np.array(result['x']), 1e-300))
                if len(full_x_log) >= max(optimize_indices) + 1:
                    best_cost  = cost
                    best_x_log = np.clip(full_x_log[optimize_indices], lb, ub)
            except Exception:
                pass
        if best_x_log is not None:
            if print_iterations:
                print(f"  ESS disk reload: found {best_cost:.6g} (pid {os.getpid()})")
            return best_cost, best_x_log
        return None

    best_x, best_fval = _run_ess(
        objective_fn,
        theta0_log,
        lb,
        ub,
        dim_refset=dim_refset,
        local_n1=local_n1,
        local_n2=local_n2,
        balance=balance,
        max_eval=max_eval,
        max_walltime_s=max_walltime_s,
        vtr=vtr,
        n_stuck_global_limit=n_stuck_global_limit,
        use_finish_polish=use_finish_polish,
        finish_polish_maxfev=finish_polish_maxfev,
        on_new_best=_on_new_best,
        inject_fn=_disk_inject if disk_inject_enabled else None,
        inject_warmup_iters=inject_warmup_iters,
        theta0_include_prob=theta0_include_prob,
        print_iterations=print_iterations,
        residuals_fn=residuals_fn,
        local_solver=local_solver,
        n_diverse_factor=n_diverse_factor,
        stagnation_perturb_sigma=stagnation_perturb_sigma,
        n_ls_children=n_ls_children,
        kick_limit=kick_limit,
        trace_list=_trace,
    )

    # Reconstruct full parameter vector and save in the standard JSON format.
    p_full = reconstruct_parameter_vector(best_x, theta0_constants_log, constant_indices)
    file_name = (
        f"{_results_dir}/{model_name} ({best_fval}) - "
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    save_result(best_fval, p_full, file_name, sim.parameter_names)

    # Save convergence trace alongside the result JSON.
    if save_trace and _trace:
        trace_file = file_name.replace('.json', '_trace.json')
        try:
            with open(trace_file, 'w') as _tf:
                json.dump(
                    {"trace": [[int(e), float(t), float(c)] for e, t, c in _trace]},
                    _tf,
                )
        except Exception as _te:
            print(f"  Could not save trace: {_te}")

    cleanup_temp_file(file_path=temp_filename)

