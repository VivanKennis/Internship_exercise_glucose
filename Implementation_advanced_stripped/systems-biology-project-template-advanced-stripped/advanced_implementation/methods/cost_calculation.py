from common.cost_functions import f_cost
from common.utils import load_best_parameters
import config


def calculate_costs(model, sims: dict, estimation_data: dict, validation_data: dict, print_individual_costs: bool = False) -> None:

    theta0 = load_best_parameters(f"./results/{model.name}", cost_key='f', model=model)

    cost_estimation = f_cost(
        theta0, sims, estimation_data, 
        config.SIMULATE_STEADY_STATE, print_costs=print_individual_costs
    )
    cost_validation = f_cost(
        theta0, sims, validation_data, 
        config.SIMULATE_STEADY_STATE, print_costs=print_individual_costs
    )

    print(f"Estimation cost: {cost_estimation}")
    print(f"Validation cost: {cost_validation}")