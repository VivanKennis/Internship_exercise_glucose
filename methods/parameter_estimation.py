from multiprocessing import cpu_count
import concurrent.futures as cf

import os

from common.parameter_estimation_functions import optimize
import config


def create_task_args(data: dict, model) -> dict:
    
    """
    Helper function to create task arguments for optimization.
    """
    
    return {
        "data"                 : data,
        "model_name"           : model.name,
        "fixed_params"         : config.FIXED_PARAMETERS,
        "simulate_steady_state": config.SIMULATE_STEADY_STATE,
        "strict_params"        : config.STRICT_BOUNDS_PARAMETERS,
        "print_iter"           : config.PRINT_ITERATIONS,
        "ess_max_eval"                : config.OptimizationConfig.ESS_MAX_EVAL,
        "ess_dim_refset"              : config.OptimizationConfig.ESS_DIM_REFSET,
        "ess_local_n1"                : config.OptimizationConfig.ESS_LOCAL_N1,
        "ess_local_n2"                : config.OptimizationConfig.ESS_LOCAL_N2,
        "ess_balance"                 : config.OptimizationConfig.ESS_BALANCE,
        "ess_max_walltime_s"          : config.OptimizationConfig.ESS_MAX_WALLTIME_S,
        "ess_vtr"                     : config.OptimizationConfig.ESS_VTR,
        "ess_n_stuck_global"          : config.OptimizationConfig.ESS_N_STUCK_GLOBAL,
        "use_finish_polish"           : config.OptimizationConfig.USE_FINISH_POLISH,
        "finish_polish_maxfev"        : config.OptimizationConfig.FINISH_POLISH_MAXFEV,
        "ess_theta0_include_prob"     : config.OptimizationConfig.ESS_THETA0_INCLUDE_PROB,
        "ess_local_solver"            : config.OptimizationConfig.LOCAL_SOLVER,
        "ess_n_diverse_factor"        : config.OptimizationConfig.ESS_N_DIVERSE_FACTOR,
        "ess_stagnation_perturb_sigma": config.OptimizationConfig.ESS_STAGNATION_PERTURB_SIGMA,
        "ess_n_ls_children"           : config.OptimizationConfig.ESS_N_LS_CHILDREN,
        "ess_kick_limit"              : config.OptimizationConfig.ESS_KICK_LIMIT,
        "ess_save_trace"              : config.OptimizationConfig.ESS_SAVE_TRACE,        
        "ess_disk_inject_enabled"     : config.OptimizationConfig.ESS_DISK_INJECT_ENABLED,
        "ess_inject_warmup_iters"     : config.OptimizationConfig.ESS_INJECT_WARMUP_ITERS,    
    }


def run_optimization(model, estimation_data: dict) -> None:
    os.makedirs(f"./results/{model.name}", exist_ok=True)

    # set up the optimization arguments
    args = create_task_args(estimation_data, model)

    cores_to_use = min(config.N_CORES, cpu_count())

    if config.OptimizationConfig.RUN_IN_PARALLEL:
        with cf.ProcessPoolExecutor(max_workers=cores_to_use) as executor:
            futures = [executor.submit(optimize, args) for _ in range(config.OptimizationConfig.N_OPTIMIZATIONS)]
            for fut in cf.as_completed(futures):
                fut.result()
    else:
        for _ in range(config.OptimizationConfig.N_OPTIMIZATIONS):
            optimize(args)
