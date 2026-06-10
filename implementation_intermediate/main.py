# standard python imports
import os
import sys
from multiprocessing import Pool
from pathlib import Path

# external library imports
import matplotlib.pyplot as plt
import sund
from scipy.stats import chi2

# project-specific imports
from functions.cost_functions import f_cost
from functions.handle_data import get_dof, load_and_process_data
from functions.parameter_estimation import optimize
from functions.plotting import plot_figures
from functions.simulation import create_sims_from_data
from functions.utils import load_best_parameters

# Model and data settings
MODEL_NAME             = "M_epinephrine15"
BASE_DIR               = Path(__file__).resolve().parent              # Define model and data file paths based on the project root for stable file paths
DATA_FILE              = BASE_DIR / "data" / "data_epinephrine.json"
VALIDATION_EXPERIMENTS = {"Kjaer 110% trained"}

# Analysis flags
DO_OPTIMIZATION       = False                       # False: no optimization
DO_PLOT               = True                        # True: show figures
SIMULATE_STEADY_STATE = True                        # True: simulation runs from -1000 to 0 before simulation starts to reach steady state
FIXED_PARAMETERS      = ["basal_nor", "basal_epi"]  # List of parameters that are fixed during optimization

N_OPTIMIZATIONS              = 5      # Number of separate optimization attempts
MAXITER                      = 500    # Maximum number of iterations per optimization
RUN_IN_PARALLEL              = True   # Run optimizations in parallel?
PRINT_ITERATIONS             = False  # Print iterations during optimization?
PERTURBATION_RATIO           = 0.3    # How much to perturb the starting values for a new optimization run
DIFFERENTIAL_EVOLUTION_RATIO = 0.7    # Probability of choosing each optimizer per run: differential evolution (70%) or dual annealing (30%)

def calculate_costs(model, sims, estimation_data, validation_data, chi2_thresholds={}, print_individual_costs=False):
    # Get best results from the results folder for the model (if saved earlier). This will be used as the starting point for the cost calculation and optimization
    results_dir = BASE_DIR / "results" / model.name 
    theta0 = load_best_parameters(str(results_dir), model)

    # Calculate and print the cost for estimation and validation data using the cost function
    cost_estimation = f_cost(
        theta0, sims, estimation_data, SIMULATE_STEADY_STATE, 
        print_costs=print_individual_costs)
    cost_validation = f_cost(
        theta0, sims, validation_data, SIMULATE_STEADY_STATE, 
        print_costs=print_individual_costs)

    print(f"\nEstimation cost: {cost_estimation:.6f}")
    print(f"Validation cost: {cost_validation:.6f}")

    # Compare costs to chi2 thresholds (if provided) to evaluate goodness of fit
    if chi2_thresholds:
        if cost_estimation < chi2_thresholds['estimation']:
            print(f"\nEstimation cost {cost_estimation:.6f} is below the chi2 threshold {chi2_thresholds['estimation']:.6f} - good fit to estimation data.")
        else:
            print(f"\nEstimation cost {cost_estimation:.6f} is above the chi2 threshold {chi2_thresholds['estimation']:.6f} - poor fit to estimation data.")

        if cost_validation < chi2_thresholds['validation']:
            print(f"Validation cost {cost_validation:.6f} is below the chi2 threshold {chi2_thresholds['validation']:.6f} - good fit to validation data.")
        else:
            print(f"Validation cost {cost_validation:.6f} is above the chi2 threshold {chi2_thresholds['validation']:.6f} - poor fit to validation data.")


def main(do_plot=True, do_optimization=False):
    #%%  Setup model and data    
            # sund.install_model relies on pathlib glob patterns and expects a relative path.
    # Install and load model
    os.chdir(BASE_DIR)
    sund.install_model(f"./models/{MODEL_NAME}.txt")
    model = sund.load_model(MODEL_NAME)

    # Load data and split into estimation and validation data
    all_data = load_and_process_data(str(DATA_FILE))
    estimation_data = {
        k: d.copy() for k, d in all_data.items() if k not in VALIDATION_EXPERIMENTS
    }
    validation_data = {
        k: d.copy() for k, d in all_data.items() if k in VALIDATION_EXPERIMENTS
    }

    # Create simulations for all experiments in the data
    sims = create_sims_from_data(model, all_data) # contains: inputs, time vectors, and model

    # Calculate chi2 thresholds for estimation and validation data based on degrees of freedom
    chi2_estimation = float(
        chi2.ppf(1 - 0.05, get_dof(estimation_data)))
    chi2_validation = float(
        chi2.ppf(1 - 0.05, get_dof(validation_data)))
    chi2_thresholds = {"estimation": chi2_estimation, "validation": chi2_validation}


    #%% Calculate the initial best solution
    print(f"Initial best solution")
    calculate_costs(
        model, sims, estimation_data, validation_data, 
        chi2_thresholds=chi2_thresholds, print_individual_costs=True)

    #%% Run optimization
    if do_optimization:
        # create folder to store results
        os.makedirs(BASE_DIR / "results" / MODEL_NAME, exist_ok=True)

        # create a dictionary of arguments to pass to the optimization function
        args = {
            "data"                 : estimation_data,
            "fixed_params"         : FIXED_PARAMETERS,
            "model_name"           : MODEL_NAME,
            "simulate_steady_state": SIMULATE_STEADY_STATE,
            "de_ratio"             : DIFFERENTIAL_EVOLUTION_RATIO,
            "max_iter"             : MAXITER,
            "perturbation_ratio"   : PERTURBATION_RATIO,
            "print_iter"           : PRINT_ITERATIONS,
        }

        if RUN_IN_PARALLEL:
            args_list = [args] * N_OPTIMIZATIONS
            with Pool(processes=os.cpu_count()) as p:
                p.map(optimize, args_list)
        else:
            for _ in range(N_OPTIMIZATIONS):
                optimize(args)
        
        # Calculate the new best solution
        print(f"Best solution")
        calculate_costs(
            model, sims, estimation_data, validation_data, 
            chi2_thresholds=chi2_thresholds, print_individual_costs=False)

    #%% Plot results
    if do_plot:
        os.makedirs(BASE_DIR / "figures", exist_ok=True)

        ## dit heb ik aangepast
        # Automatically collect all unique figure numbers from data
        fig_numbers = sorted({
            obs["plotting_info"]["fig"]
            for d in all_data.values()
            for obs in d["Observables"].values()
        })
        fig_names = {fig_num: f"Figure_{fig_num}" for fig_num in fig_numbers}
        ## tot hier

        theta = load_best_parameters(str(BASE_DIR / "results" / MODEL_NAME), model)
        all_data = {**estimation_data, **validation_data}

        plot_figures(theta, sims, all_data, model, fig_names, SIMULATE_STEADY_STATE)
        plt.show()



if __name__ == "__main__":
    # Override default analysis flags from command-line arguments.
    do_plot         = True if "--plot" in sys.argv else DO_PLOT
    do_optimization = True if "--opt"  in sys.argv else DO_OPTIMIZATION

    main(do_plot=do_plot, do_optimization=do_optimization)