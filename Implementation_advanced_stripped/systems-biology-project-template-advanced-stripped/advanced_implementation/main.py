# define imports
from methods.cost_calculation import calculate_costs
from methods.parameter_identifiability import run_PI_analysis
from methods.plotting import plot_results
from methods.reverse_PPL import run_reverse_PPL_analysis
from methods.parameter_estimation import run_optimization
from methods.setup import setup_model_and_data

# Import configuration settings
import config

def main(
    DO_PLOT: bool=True, DO_OPT: bool=False,
    DO_PI: bool=False, DO_REVERSE_PPL: bool=False, 
    PLOT_ONLY: bool=False):
    
    # Setup model and data
    (model, sims, estimation_data, validation_data, 
        chi2_thresholds) = setup_model_and_data(config.MODEL_NAME)

    # Calculate the initial best solution
    if not PLOT_ONLY:
        print(f"Initial best solution")
        calculate_costs(model, sims, estimation_data, validation_data)

    # Run optimization
    if DO_OPT and not PLOT_ONLY:
        run_optimization(model, estimation_data)

        # Calculate the new best solution
        print(f"Best solution")
        calculate_costs(model, sims, estimation_data, validation_data)

    # Run parameter identifiability analysis
    if DO_PI and not PLOT_ONLY:
        chi2_limit = chi2_thresholds["estimation"]
        run_PI_analysis(model, estimation_data, chi2_limit)

    # Run reverse prediction profile likelihood analysis
    if DO_REVERSE_PPL and not PLOT_ONLY:
        all_data = {**estimation_data, **validation_data}
        chi2_limit = chi2_thresholds["estimation"]
        validation_experiments = set(validation_data.keys())
        run_reverse_PPL_analysis(model, sims, all_data, validation_experiments, chi2_limit)

    if DO_PLOT or PLOT_ONLY:
        plot_results(
            model, sims, estimation_data, validation_data,
            plot_pi=DO_PI or PLOT_ONLY)

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
Usage: python main.py [OPTIONS]

Options:
    --plot           Plot results after running analyses. Saves figures to the
                    figures/ directory.

    --plot-only      Skip all computation and only (re)plot from existing results
                    already saved in the results/ directories. Useful for
                    adjusting figures without re-running analyses.

    --opt            Run parameter estimation. Uses a numerical optimizer to find
                    parameter values that best fit the experimental data.
                    Results are saved to results/.

    --pi             Run parameter identifiability analysis. Checks whether each
                    parameter can be uniquely determined from the available data,
                    by systematically fixing one parameter at a time and
                    re-optimizing. Results are saved to results_PI/.

    --reverse-ppl    Run reverse prediction profile likelihood analysis.
                    Explores prediction uncertainty by directly optimizing
                    over prediction values rather than stepping along profiles.
                    Results are saved to results_reverse_PPL/.

    -h, --help       Show this help message and exit.

If no options are provided, the flags defined in config.py are used instead.
Multiple options can be combined, e.g.:
    python main.py --opt --parameter-pl --plot
""")
        sys.exit(0)

    # Check for command-line arguments to override config settings
    DO_PLOT         = True if "--plot"         in sys.argv else config.DO_PLOT
    DO_OPT          = True if "--opt"          in sys.argv else config.DO_PARAMETER_ESTIMATION
    DO_PI           = True if "--pi"           in sys.argv else config.DO_PARAMETER_IDENTIFIABILITY
    DO_REVERSE_PPL  = True if "--reverse-ppl"  in sys.argv else config.DO_REVERSE_PPL
    PLOT_ONLY       = True if "--plot-only"    in sys.argv else config.DO_PLOT_ONLY

    main(
        DO_PLOT=DO_PLOT, DO_OPT=DO_OPT, 
        DO_PI=DO_PI, DO_REVERSE_PPL=DO_REVERSE_PPL,
        PLOT_ONLY=PLOT_ONLY
    )