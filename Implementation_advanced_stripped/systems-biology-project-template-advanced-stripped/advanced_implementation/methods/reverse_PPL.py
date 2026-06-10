import os
import random
import time
from multiprocessing import Pool

import numpy

from common.collect_model_uncertainty import collect_reverse_PPL, setup_reverse_PPL
from common.uncertainty import optimize_reverse_PPL
import config


def create_task_args(data: dict, model, chi2_limit: float, exp: str, obs: str, time: float, direction: int = 1) -> dict:
    
    """
    Helper function to create task arguments for optimization.
    """

    return {
        "chi2_limit": chi2_limit,
        "data": data,
        "dir": direction,
        "fixed_params": config.FIXED_PARAMETERS,
        "model_name": model.name,
        "simulate_steady_state": config.SIMULATE_STEADY_STATE,
        "strict_params": config.STRICT_BOUNDS_PARAMETERS,
        "exp": exp,
        "obs": obs,
        "time": time,
        "de_ratio": config.ReversePPLConfig.DIFFERENTIAL_EVOLUTION_RATIO,
        "max_iter": config.ReversePPLConfig.MAXITER,
        "perturbation_ratio": config.ReversePPLConfig.PERTURBATION_RATIO,
        "pop_size": config.ReversePPLConfig.INIT_POP_SIZE,
        "print_iter": config.PRINT_ITERATIONS,
    }


def run_reverse_PPL_analysis(model, sims: dict, all_data: dict, validation_experiments: set[str], chi2_estimation: float) -> None:

    # create results folder
    os.makedirs(f"./results_reverse_PPL/{model.name}", exist_ok=True)

    # set up the reverse PPL analysis
    ALL_DATA_reverse_PPL, reverse_PPL_combinations = setup_reverse_PPL(
        model, sims, all_data, chi2_estimation, validation_experiments, config.SIMULATE_STEADY_STATE)

    # set the seed for each process to avoid getting the same random numbers
    numpy.random.seed(int(time.time()) + os.getpid())

    # shuffle the combinations
    shuffled_reverse_PPL_items = list(reverse_PPL_combinations.items())
    random.shuffle(shuffled_reverse_PPL_items)

    # populate the args: N_OPTIMIZATIONS restarts × each combination × both directions
    args_list = []
    for _ in range(config.ReversePPLConfig.N_OPTIMIZATIONS):
        for _, reverse_PPL_opt in shuffled_reverse_PPL_items:
            for dir in [-1, 1]:
                args_reverse_PPL = create_task_args(
                    ALL_DATA_reverse_PPL, model, chi2_estimation, reverse_PPL_opt["k_exp"], reverse_PPL_opt["k_obs"], reverse_PPL_opt["Time"],
                    direction=dir)
                args_list.append(args_reverse_PPL)

    # run the method in parallel
    if config.ReversePPLConfig.RUN_IN_PARALLEL:
        with Pool(processes=config.N_CORES) as p:
            p.map(optimize_reverse_PPL, args_list)

    # run method sequentially
    else:
        for args_reverse_PPL in args_list:
            optimize_reverse_PPL(args_reverse_PPL)

    # collect reverse PPL results for plotting
    collect_reverse_PPL(model, sims, all_data)