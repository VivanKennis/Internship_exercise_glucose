## A Mechanistic Model of Blood Glucose Regulation during Exercise Incorporating Epinephrine and Norepinephrine Dynamics
This project is based on a template developed for modelling activities, originally from:

Henrik Podéus, Gustav Magnusson, Sasan Keshmiri, Kajsa Tunedal, Nicolas Sundqvist, William Lövfors, and Gunnar Cedersund. *SUND: simulation using nonlinear dynamic models - a toolbox for simulating multi-level, time-dynamic systems in a modular way.* 10 2025.

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

#### Main script
The `main.py` script is the entry point of the project. It calls and executes the tasks listed in its headers by using the corresponding functions from the `methods` folder, based on the settings specified in `config`. The created figures are saved in the `figures` folder. 

#### Config
The `config` file contains all the `user-configurable settings` of the project, such as the model, dataset(s), and options used for the different tasks. These settings are read and used by `main.py`.

#### Plot datasets
The `plot_datasets.py` script generates the figures in the `Datasets` folder, showing all variables in the model together with the data for each dataset.

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

#### Models
There is also a `models` folder containing three model files:

- **`M_lactate3.txt`**: model for lactate only.
- **`M_epinephrine23.txt`**: model for lactate, norepinephrine, and epinephrine.
- **`M_glucose10.txt`**: model for lactate, norepinephrine, epinephrine, insulin, and glucose.

This folder also contains an `old_models` subfolder with older versions of these models.

#### Results
The `results` folder contains the results of the parameter estimation. It contains a separate subfolder for each model, as well an `old_models` subfolder for the older model versions. The result files are named after the model, followed by the cost of the corresponding parameter set and the date and time at which the parameter estimation was run. Each file contains the cost, the resulting parameter values, and their names.

#### Figures
Additionally, there is a `figures` directory containing five subfolders; 

- **`M_epinephrine23`**: contains the result figures for model `M_epinephrine23`. For the variables lactate, norepinephrine, and epinephrine, there is a figure showing, for each dataset containing that variable, a plot with the data and the corresponding model fit.
- **`M_glucose10`**: same as above, but for the variables lactate, norepinephrine, epinephrine, insulin, and glucose.
- **`M_lactate3`**: same as above, but for lactate only.
- **`Datasets`**: contains a separate figure for each dataset, showing all variables in the model together with the data.
- **`old_models`**: contains the figures for the older model versions.

### The data structure
There are 5 data files:

- **`data_lactate`**: contains only the data for lactate.
- **`data_epinephrine`**: contains the data for lactate, norepinephrine, and epinephrine.
- **`data_glucose`**: contains the data for lactate, norepinephrine, epinephrine, insulin, and glucose.
- **`data_verification`**: identical to `data_glucose`, but with the intensity set to 0 everywhere, used for verification.
- **`data_format`**: shows the structure according to the format and serves as an example. The content itself is not used, only the structure. 

All data files contain multiple levels. The format structure is described below:

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
