import sund
from scipy.stats import chi2

from common.handle_data import get_dof, load_and_process_data
from common.simulation import create_sims_from_data
import config


def load_and_install_model(model_name: str, install_model: bool = True):
    # Load and setup model
    if install_model:
        sund.install_model(f"./models/{model_name}.txt")
    model = sund.load_model(model_name)
    
    return model


def setup_model_and_data(model_name: str, install_model: bool = True):
    model = load_and_install_model(model_name, install_model)

    # Load and process data
    all_data = load_and_process_data(config.DATA_FILE)
    estimation_data = {k: d.copy() for k, d in all_data.items() if k not in config.VALIDATION_EXPERIMENTS}
    validation_data = {k: d.copy() for k, d in all_data.items() if k in config.VALIDATION_EXPERIMENTS}

    # Create simulations
    sims = create_sims_from_data(model, all_data)

    # Calculate chi2 thresholds
    chi2_estimation = float(chi2.ppf(1 - config.CHI2_SIGNIFICANCE_LEVEL, get_dof(estimation_data)))
    chi2_validation = float(chi2.ppf(1 - config.CHI2_SIGNIFICANCE_LEVEL, get_dof(validation_data)))
    chi2_thresholds = {"estimation": chi2_estimation, "validation": chi2_validation}
    
    if config.PRINT_DOF:
        print(f'Estimation DOF = {get_dof(estimation_data)}\nchi2-cutoff = {chi2_estimation}')
        print(f'Validation DOF = {get_dof(validation_data)}\nchi2-cutoff = {chi2_validation}')
    
    return model, sims, estimation_data, validation_data, chi2_thresholds