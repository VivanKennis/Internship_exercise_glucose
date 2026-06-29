"""
Configuration file for systems biology project.

This file contains all user-configurable settings for the project.
When starting a new project, modify the settings in this file to match your needs.
"""

# ============================================================================
# MODEL AND DATA SETTINGS
# ============================================================================

# Name of the model file (without .txt extension)
# The model file should be located in the ./models/ directory
MODEL_NAME: str = "M_epinephrine18" #"M_epinephrine16"

# Set to True if the model needs to simulate a steady state before running experiments
SIMULATE_STEADY_STATE: bool = True

# ============================================================================
# ANALYSIS FLAGS - Control which analyses to run
# ============================================================================

# Set to True to run parameter optimization
DO_PARAMETER_ESTIMATION: bool = True

# Set to True to run parameter identifiability analysis
DO_PARAMETER_IDENTIFIABILITY: bool = False

# Set to True to run reverse prediction profile likelihood analysis
DO_REVERSE_PPL: bool = False

# Set to True to plot results
DO_PLOT: bool = True

# Set to True to skip all computation and only (re)plot from existing results
DO_PLOT_ONLY: bool = False

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Path to the data file
DATA_FILE: str = "./data/data_epinephrine.json"

# Set to True to print degrees of freedom during data loading
PRINT_DOF: bool = False

# Define which experiments are used for validation (not optimization)
# Example: {"Experiment3", "Experiment4"}
VALIDATION_EXPERIMENTS: set = {"Kjaer 110% trained", "Gaitanos 100%"}

# Significance level for chi-square thresholds
CHI2_SIGNIFICANCE_LEVEL: float = 0.05

# Number of CPU cores available for parallel methods
N_CORES: int = 4

# Set to True to print iteration/step progress for all methods
PRINT_ITERATIONS: bool = True

# ============================================================================
# PARAMETER CONSTRAINTS
# ============================================================================

# List of parameter names that should be fixed during optimization
# These parameters will not be changed during the optimization process
# Example: ["k1", "k2", "kfeed"]
FIXED_PARAMETERS: list = ["vmax_thresh", "km_thresh", "n_thresh", "v_cons", "km_O2", "n_O2", "k_restore", "vmax_lac", "km_lac", "n_lac", "elim_lactate"]

# List of parameter names that should be optimized within strict bounds
# These parameters will be constrained to their defined bounds more strictly
# Example: ["k1", "k2", "kfeed"]
STRICT_BOUNDS_PARAMETERS: list = []

# ============================================================================
# OPTIMIZATION SETTINGS
# ============================================================================

class OptimizationConfig:
    """Configuration for standard parameter optimization."""
    
    # Number of independent optimization runs to perform
    N_OPTIMIZATIONS: int = 3
    
    # Maximum number of iterations per optimization
    MAXITER: int = 100
    
    # Set to True to run optimizations in parallel (faster but uses more CPU)
    RUN_IN_PARALLEL: bool = True

    # -------------------------------------------------------------------------
    # ESS (Enhanced Scatter Search) settings
    # -------------------------------------------------------------------------

    # Maximum number of objective function evaluations per ESS run.
    # The actual limit is max(ESS_MAX_EVAL, 1000 * n_params).
    ESS_MAX_EVAL: int = 10_000

    # Size of the ESS reference set (RefSet).
    # Larger = more diversity but more evaluations per iteration.
    # Auto-scaled up to max(5, ceil((1 + sqrt(4n)) / 2)) if this is too small.
    ESS_DIM_REFSET: int = 20

    # Minimum iterations before first local search.
    ESS_LOCAL_N1: int = 1

    # Minimum iterations between consecutive local searches.
    ESS_LOCAL_N2: int = 2

    # Quality vs. diversity balance for local search starting points.
    # 0 = focus on best (quality), 1 = focus on unexplored (diversity).
    ESS_BALANCE: float = 0.5

    # Maximum wall-clock time per ESS run in seconds.
    # None = no limit; useful on HPC clusters with fixed job time slots.
    ESS_MAX_WALLTIME_S: float | None = None

    # Value-to-reach: stop as soon as the best cost drops to or below this value.
    # None = disabled.
    ESS_VTR: float | None = None

    # Global stagnation stopping: stop after this many consecutive iterations
    # without any improvement to the global best.  0 = disabled.
    ESS_N_STUCK_GLOBAL: int = 300

    # Set to True to run a final local polish on the overall best solution.
    USE_FINISH_POLISH: bool = True

    # Minimum function evaluations for the final finish polish step.
    # The actual limit is max(FINISH_POLISH_MAXFEV, 500 * n_params).
    FINISH_POLISH_MAXFEV: int = 5000
    
    # Probability that the best-known parameter set (theta0) is included in
    # the initial diverse set when seeding the RefSet. 1.0 = always include
    # theta0, 0.0 = never include (pure LHS exploration).
    ESS_THETA0_INCLUDE_PROB: float = 0.5

    # Local search backend used inside _run_ess.
    # "nelder-mead" — Nelder-Mead (no extra dependencies required)
    # "dfo-ls"      — DFO-LS (derivative-free, uses residual vector; install dfo-ls first)
    LOCAL_SOLVER: str = "dfo-ls"

    # Scaling factor for the initial diverse-set size used to seed the RefSet.
    # n_diverse = max(factor * n_params, factor * dim_refset).
    # Larger values improve basin coverage at the cost of more evaluations at startup.
    ESS_N_DIVERSE_FACTOR: int = 20

    # Standard deviation (as fraction of bound width) for the perturbation-of-best
    # half of stagnation resets.  The other half still draws fresh LHS points.
    ESS_STAGNATION_PERTURB_SIGMA: float = 0.1

    # Maximum number of local searches launched per ESS iteration.
    # 0 = auto-scale: min(3, max(1, dim_refset // 4)).
    ESS_N_LS_CHILDREN: int = 0

    # Number of global-stagnation kicks before a hard stop.
    # A kick replaces the bottom half of the RefSet with LHS-around-best samples
    # (sigma = 0.5*(ub-lb)), resetting the stagnation counter each time.
    ESS_KICK_LIMIT: int = 3

    # Whether to save the (n_evals, elapsed_s, cost) convergence trace
    # alongside the final optimisation JSON (as a separate *_trace.json file).
    ESS_SAVE_TRACE: bool = False

    # Enable/disable disk-inject (sharing best solutions across parallel workers).
    # Disable to keep workers fully independent; useful for A/B testing or when
    # monoculture / local-minimum trapping is suspected.
    ESS_DISK_INJECT_ENABLED: bool = False

    # Number of ESS iterations each worker explores independently before
    # accepting any injected solution from disk.  Prevents early monoculture.
    ESS_INJECT_WARMUP_ITERS: int = 75

# ============================================================================
# PARAMETER IDENTIFIABILITY SETTINGS
# ============================================================================

class ParameterIdentifiabilityConfig:
    """Configuration for parameter identifiability (PI) analysis."""
    
    # Number of independent PI optimization runs
    # Should be an even number (half maximize, half minimize each parameter)
    N_OPTIMIZATIONS: int = 50
    
    # Maximum number of iterations per PI optimization
    MAXITER: int = 50
    
    # Set to True to run PI optimizations in parallel
    RUN_IN_PARALLEL: bool = True

    # Ratio of perturbations in the PI optimization
    PERTURBATION_RATIO: float = 0.3
    
    # Ratio of differential evolution in the PI optimization
    DIFFERENTIAL_EVOLUTION_RATIO: float = 0.7
    
    # Size of the initial population for PI optimization
    INIT_POP_SIZE: int = 100


# ============================================================================
# REVERSE PPL SETTINGS
# ============================================================================

class ReversePPLConfig:
    """Configuration for reverse prediction profile likelihood analysis."""
    
    # Maximum number of iterations per optimization
    MAXITER: int = 50
    
    # Set to True to run optimizations in parallel
    RUN_IN_PARALLEL: bool = True

    # Ratio of perturbations in the optimization
    PERTURBATION_RATIO: float = 0.3
    
    # Ratio of differential evolution in the optimization
    DIFFERENTIAL_EVOLUTION_RATIO: float = 0.7
    
    # Size of the initial population for optimization
    INIT_POP_SIZE: int = 100
    
    # Direction for optimization (1 or -1)
    # This is typically set automatically, but can be overridden here
    DIRECTION: int = 1

    # Number of optimization restarts per (exp, obs, t, direction) combination.
    # Increase this to escape local optima at the cost of more computation time.
    N_OPTIMIZATIONS: int = 1


"""
QUICK START GUIDE:
------------------
1. Set MODEL_NAME to your model file name (without .txt extension)
2. Update VALIDATION_EXPERIMENTS to specify which experiments are for validation
3. Set the DO_* flags to control which analyses to run
4. For optimization: adjust OptimizationConfig.N_OPTIMIZATIONS and MAXITER
5. For parallel execution: set RUN_IN_PARALLEL = True in the relevant config class

PARAMETER CONSTRAINTS:
---------------------
- FIXED_PARAMETERS: Parameters that won't change during optimization
- STRICT_BOUNDS_PARAMETERS: Parameters with strict boundary constraints

PERFORMANCE TIPS:
----------------
- Start with low N_OPTIMIZATIONS and MAXITER for testing
- Enable RUN_IN_PARALLEL for faster computation (requires multiple CPU cores)
- Increase N_OPTIMIZATIONS for more thorough parameter space exploration
- Adjust INIT_POP_SIZE based on the number of parameters (typically 10-20x the number of parameters)
"""
