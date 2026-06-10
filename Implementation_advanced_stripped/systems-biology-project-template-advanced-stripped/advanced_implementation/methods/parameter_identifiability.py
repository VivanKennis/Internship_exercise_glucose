import json
import os
from multiprocessing import Pool
from typing import Any

from common.identifiability import optimize_PI
from common.utils import NumpyArrayEncoder, load_best_parameters
import config


def create_task_args(data: dict, model, chi2_limit: float, direction: int = 1) -> dict:
    
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
        "de_ratio": config.ParameterIdentifiabilityConfig.DIFFERENTIAL_EVOLUTION_RATIO,
        "max_iter": config.ParameterIdentifiabilityConfig.MAXITER,
        "perturbation_ratio": config.ParameterIdentifiabilityConfig.PERTURBATION_RATIO,
        "pop_size": config.ParameterIdentifiabilityConfig.INIT_POP_SIZE,
        "print_iter": config.PRINT_ITERATIONS,
    }


def setup_PI(model: Any) -> None:
    # The best parameters found so far with respect to the agreement to data
    theta_best = load_best_parameters(
        f"./results/{model.name}", cost_key='f', model=model)

    PI_foldername = f"./results_PI/{model.name}"
    
    # If the current model has not been tested, save an initial guess for all parameters
    if not os.path.exists(PI_foldername) or len(os.listdir(PI_foldername)) == 0:
        # make sure the results folder exists
        os.makedirs(PI_foldername, exist_ok=True)
        for p_idx, p_name in enumerate(model.parameter_names):
            try:
                with open(f"{PI_foldername}/{model.name}-{p_name}-initial-({theta_best[p_idx]:e}).json", 'w') as file:
                    out = {"f": theta_best[p_idx], "x": theta_best}
                    json.dump(out, file, cls=NumpyArrayEncoder)
            except PermissionError:
                print(
                    f"Permission denied for file '{PI_foldername}/{model.name}-{p_name}-initial-({theta_best[p_idx]}).json'. Not saving current solution.")


def summarize_PI_results(model_name: str) -> None:
    """
    Read all PI result JSON files from results_PI/{model_name}/ and write a
    summary of the identified min/max bound per parameter to
    results_PI/{model_name}/PI_summary.json.

    Summary structure:
        {
            "k1": {"min": <float>, "max": <float>,
                   "min_x": [...], "max_x": [...]},
            ...
        }
    """
    folder = f"./results_PI/{model_name}"
    if not os.path.exists(folder):
        return

    # group results by parameter name
    # filename pattern: {model_name}-{param_name}-... .json
    prefix = f"{model_name}-"
    groups: dict[str, list[dict]] = {}

    for fname in os.listdir(folder):
        if not fname.endswith(".json") or fname == "PI_summary.json":
            continue
        # strip model prefix
        rest = fname[len(prefix):]
        # extract param name — everything up to the first '-' after the prefix
        param_name = rest.split("-")[0]
        if not param_name:
            continue
        try:
            with open(os.path.join(folder, fname)) as fh:
                data = json.load(fh)
            if "f" in data and "x" in data:
                groups.setdefault(param_name, []).append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not groups:
        return

    summary: dict[str, dict] = {}
    for param_name, entries in groups.items():
        f_values = [e["f"] for e in entries]
        min_entry = min(entries, key=lambda e: e["f"])
        max_entry = max(entries, key=lambda e: e["f"])
        summary[param_name] = {
            "min": float(min_entry["f"]),
            "max": float(max_entry["f"]),
            "min_x": [float(v) for v in min_entry["x"]],
            "max_x": [float(v) for v in max_entry["x"]],
        }

    out_path = os.path.join(folder, "PI_summary.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, cls=NumpyArrayEncoder)
    print(f"PI summary saved to {out_path}")


def run_PI_analysis(model, estimation_data: dict, chi2_limit: float) -> None:

    setup_PI(model)

    args_list = [
        create_task_args(
            estimation_data, model, chi2_limit,
            direction=1 if i % 2 == 0 else -1)
        for i in range(config.ParameterIdentifiabilityConfig.N_OPTIMIZATIONS)
    ]
    
    # run optimization in parallel
    if config.ParameterIdentifiabilityConfig.RUN_IN_PARALLEL:
        with Pool(processes=config.N_CORES) as p:
            p.map(optimize_PI, args_list)

    # run optimization sequentially
    else:
        for arg in args_list:
            optimize_PI(arg)

    summarize_PI_results(model.name)
