import numpy as np

from functions.simulation import simulate_model, simulate_model_steady_state
from functions.utils import reconstruct_parameter_vector


def f_cost(p, sims, data, simulate_steady_state = False, print_costs = False):
    cost = 0
    # k_exp = experiment name, d = dictionary with data for that experiment
    for k_exp, d in data.items():
        try:    # probeer de simulatie uit te voeren, als er een fout optreed dan door naar "except"
            # Get the simulation object for the current experiment, reset model states to initial conditions, and simulate the model
            sim = sims[k_exp]
            ic = sim.state_values
            sim.reset_states()  # reset zodat de simulatie niet verder gaat waar de vorige simulatie is geëindigd, maar weer bij de beginwaarden van de staten begint
            
            if simulate_steady_state:
                simulate_model_steady_state(sim, ic, p)
                ic = sim.state_values

            simulate_model(sim, ic, p, time_vector=data[k_exp]["all_times"])

            # voor elke observable in de data, zoekt de bijbehorende simulatie output (zodat datapunt en simulatiepunt op hetzelfde tijdstip zijn)
            for k_obs, obs in d["Observables"].items():
                idx = sim.feature_names.index(k_obs)
                y_sim = sim.feature_data[:, idx]

                # bereken de chi2 cost voor deze observable en tel deze op bij de totale cost
                y_sim = y_sim[np.searchsorted(sim.time_vector, obs["Time"])]
                cost += np.square((obs['Mean']-y_sim)/obs['SEM']).sum()

                if print_costs:
                    c = np.square((obs['Mean']-y_sim)/obs['SEM']).sum()
                    print(f"{k_exp}-{k_obs}: {c}")

        # als er een fout optreed dan een hoge cost geven
        except Exception as e:
            if "CVODE" not in str(e):
                print(f"Error in {k_exp}: {e}")
            cost += 1e20
            return cost

    return cost


def f_cost_log_with_fixed_parameters(p_log, sims, data, simulate_steady_state = False, 
                                     theta0_constant = np.array([]), constant_indices = []):
    # Reconstruct the full parameter vector by interleaving optimized and constant parameters
    p_reconstructed = reconstruct_parameter_vector(
        p_log, theta0_constant, constant_indices)
    
    # Calculate the cost using the original cost function
    p = np.exp(p_reconstructed)
    cost = f_cost(p, sims, data, simulate_steady_state)

    return cost
