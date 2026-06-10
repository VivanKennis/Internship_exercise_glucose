## This project presents template code solution for our possible modelling activities
The project contains a main script `main.py`, with code blocks for the headers listed below, that calls and execute the task from separate functions in the "methods/" folder.

All needed packages can be installed through:

### uv

```bash
uv sync
```

### pip

```bash
pip install -r requirements.txt
``` 

The showcased methods are listed below:

- Importing and setting up a sund model object
- Setting up a sund simulation object using the `.json` data structure found under "data/"
- Calculating the cost from a generic objective function
- Generic plot function using the data structure
- Optimization using eSS + Scipy minimize
- Parameter Identifiability (PI) analysis using Scipy
- Parameter Profile Likelihood (PL) using Scipy
- Prediction Profile Likelihood (PPL) using Scipy
- Reverse Prediction Profile Likelihood (reverse PPL) using Scipy
- Markov Chain Monte Carlo (MCMC) sampling

### The code structure

#### Functions
The code is divided into different files in the `common` folder in the following general way: 

* handle_data.py: Useful functions for loading the data set, as well as to get the degrees of freedom
* simulation.py: Functions used to create the simulation objects
* cost_functions.py: The general cost functions used to evaluate parameter sets, either in linear or log-space
* parameter_estimation.py: The scripts needed to do parameter estimation using optimization algorithms 
* plotting.py: Functions used for plotting the simulations and data
* model_uncertainty.py: Functions used to estimate the model uncertainty
* identifiability.py: Functions used for parameter identifiability and profile likelihood analysis
* uncertainty.py: Functions used for uncertainty quantification including MCMC sampling
* format_json.py: Custom JSON formatting for saving results with readable numeric precision
* utils.py: General functions that might be useful (e.g. formatting numpy arrays to a json compliant format before saving)

#### Methods
In the `methods` folder you will find common methods prepared for you, using the functions defined in the `common` folder:

* setup.py: prepares the model and data
* cost_calculation.py: simulates the model and calculates the cost
* plotting.py: sets up and plots the model results
* parameter_estimation.py: runs a parameter estimation (parameter optimization) of your model
* parameter_identifiability.py: runs parameter identifiability (PI) analysis of your parameters
* parameter_profile_likelihood.py: runs a parameter profile likelihood (PL) analysis of your parameters
* prediction_profile_likelihood.py: runs a prediction profile likelihood (PPL) analysis of your simulation uncertainty in every time point
* reverse_PPL.py: runs a reverse prediction profile likelihood analysis by directly optimizing over prediction values
* mcmc.py: runs Markov Chain Monte Carlo (MCMC) sampling for uncertainty quantification

### The data structure
To make the code more generic and reusable, we utilize a data structure to make the "special case solutions" less frequent. This is not a universal solution, but it works well for our typical systems. The structure is described below:

- Level 1: Experiment - often a study that has generated data from an experimental protocol.
    - Level 2: meta - [optional] allows you to attach meta information to the data experiment.
    - Level 2: extra - [optional] allows you to leave comments or similar about the experiment.
    - Level 2: input - describes the experimental protocol that affects the model.
        - Level 3: Name of input 1 - the name of a input that maps to a input in a sund model file.
            - Level 4: Time values - a list of time values for when the input change values. Corresponds to the sund.Activity objects. 
            - Level 4: Input values - a list with the values that the input becomes at the corresponding time value.
            - Level 4: type or mode - a string of the sund activity type or state manipulation mode this input should have.
            - Level 4 (if mode): name - a string detailing which state name should be affected by the manipulation.
    - Level 2: Observables - lists the different observables measured during the experimental protocol. 
        - Level 3: Name of observable - the name of a observable. The name maps to a model feature in a sund model file.
            - Level 4: Time - a list of time values where measurements were taken. 
            - Level 4: Mean - a list of mean values of the measurements. 
            - Level 4: SEM - a list of SEM values of the measurements. 
            - Level 4: Points - [optional] a list of lists of the measured values from an individual in the study. 
            - Level 4: unit - [optional] the unit of the observable. 
            - Level 4: plotting_info - contains information specifying the plotting rules. 
                - Level 5: fig - specify the figure number.
                - Level 5: subplot_idx - specify the subplot index.
                - Level 5: shape - specify the shape, number of rows and number of columns, in the figure ([rows, columns]).
                - Level 5: plot_colors - specifies the rgb color codes (triplets) to use when plotting as [[plot color], [errorbar color], [errorbar fill color]].
                - Level 5: marker - specifies which marker the experimental should be plotted with.
                - Level 5: title" - specifies the title of the subplot.
                - Level 5: xlabel - specifies the label of the x-axis.
                - Level 5: ylabel - specifies the label of the y-axis.
                - Level 5: xlim": - specifies the limits of the x-axis.
                - Level 5: ylim": - specifies the limits of the y-axis

```
{
    "Experiment1": {
        "meta": {
            "doi": "doi.org/xyz123",
            "Published": 1990
        },
        "extra": {
            "Quality": "The data was very noisy."
        },
        "input": {
            "inp1": {
                "t": [-Infinity, 5, 6],
                "f": [0, 1, 0],
                "type: "piecewise_constant"
            }, 
            "inp2": {
                "t": [-Infinity],
                "f": [1],
                "type": "constant"
            }
        },
        "Observables": {
            "Obs1": {
                "Time": [0, 5, 10],
                "Mean": [1, 2, 3],
                "SEM": [0.02, 0.02, 0.03],
                "unit": "nM",
                "plotting_info": {
                    "fig": 1,
                    "subplot_idx": 1,
                    "shape": [2, 2],
                    "plot_colors": [
                        [102, 153, 255],
                        [102, 153, 255],
                        [102, 153, 255]
                        ],
                    "marker": "o",
                    "title": "Experiment 1 - Phosphorylated R",
                    "xlabel": "Time (minutes)",
                    "ylabel": "Rp (fraction)",
                    "ylim": [0, 0.2],
                    "xlim": [0, 60]
                }
            },
            "Obs2": {
                "Time": [0, 5, 10],
                "Mean": [0, 0, 0.15112563],
                "SEM": [Infinity, Infinity, 0.004],
                "unit": "nM",
                "plotting_info": {
                    "fig": 1,
                    "subplot_idx": 3,
                    "shape": [2, 2],
                    "plot_colors": [
                        [102, 153, 255],
                        [102, 153, 255],
                        [102, 153, 255]
                        ],
                    "marker": "o",
                    "title": "Experiment 1 - Phosphorylated S",
                    "xlabel": "Time (minutes)",
                    "ylabel": "Sp (fraction)",
                    "ylim": [0, 0.3],
                    "xlim": [0, 60]
                }
            }
        }
    },
    "Experiment2": {
        "input": {
            "inp1": {
                "t": [5, 60],
                "f": [1, 2],
                "type": "mode",
                "name": "A"
            }
        },
        "Observables": {
            "Obs1": {
                "Time": [0, 65, 70, 75, 80, 85, 90, 95, 100],
                "Mean": [0, 0.130148, 0.090192, 0.059823, 0.04701, 0.034251, 0.032175, 0.029674, 0.026648],
                "SEM": [Infinity, 0.022427, 0.012864, 0.009188, 0.008876, 0.008798, 0.008431, 0.008488, 0.008232],
                "Points": [
                    [0, 0.180148, 0.140192, 0.109823, 0.06701, 0.054251, 0.052175, 0.049674, 0.046648],
                    [0, 0.100148, 0.080192, 0.039823, 0.03701, 0.033251, 0.031175, 0.028674, 0.025648],
                    [0, 0.080148, 0.060192, 0.029823, 0.02701, 0.024251, 0.022175, 0.019674, 0.016648]
                    ],
                "unit": "nM",
                "plotting_info": {
                    "fig": 2,
                    "subplot_idx": 1,
                    "shape": [1, 2],
                    "plot_colors": [[255, 102, 102], [0, 0, 0], "None"],
                    "marker": "o",
                    "title": "Experiment 3 - Phosphorylated R",
                    "xlabel": "Time (minutes)",
                    "ylabel": "Rp (fraction)",
                    "ylim": [0, 0.2],
                    "xlim": [0, 110]
                }
            }
        }   
    }
}

```