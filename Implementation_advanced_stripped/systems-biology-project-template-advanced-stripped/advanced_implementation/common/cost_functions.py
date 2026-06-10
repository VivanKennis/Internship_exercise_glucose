import numpy as np
from scipy.integrate import trapezoid

from common.simulation import simulate_model, simulate_model_steady_state
from common.utils import reconstruct_parameter_vector

ROUNDING_PRECISION = 15 # offset pythons default rounding error of 2.2e-16 when exp()


def f_cost(p: np.ndarray, sims: dict, data: dict, simulate_steady_state: bool = False, print_costs: bool = False) -> float:
    cost = 0
    for k_exp, d in data.items():
        try:
            sim = sims[k_exp]
            ic = sim.state_values
            main_time_vector = np.array(sim.time_vector, copy=True)
            sim.reset_states()
            if simulate_steady_state:
                simulate_model_steady_state(sim, ic, p)
                ic = sim.state_values

            simulate_model(sim, ic, p, time_vector=main_time_vector)

            for k_obs, obs in d["Observables"].items():
                idx = sim.feature_names.index(k_obs)
                y_sim = sim.feature_data[:, idx]

                obs_idx = np.searchsorted(sim.time_vector, obs["Time"])
                obs_idx = np.clip(obs_idx, 0, len(sim.time_vector) - 1)
                y_sim = y_sim[obs_idx]
                cost += np.square((obs['Mean']-y_sim)/obs['SEM']).sum()

                if print_costs:
                    c = np.square((obs['Mean']-y_sim)/obs['SEM']).sum()
                    print(f"{k_exp}-{k_obs}: {c}")

        except Exception as e:
            if "CVODE" not in str(e):
                print(f"Error in {k_exp}: {e}")
            cost += 1e20
            return cost

    return cost


def f_cost_log(p_log: np.ndarray, sims: dict, data: dict, simulate_steady_state: bool = False) -> float:
    p = np.exp(p_log).round(ROUNDING_PRECISION)
    cost = f_cost(p, sims, data, simulate_steady_state=simulate_steady_state)

    return cost


def f_cost_log_with_fixed_parameters(p_log: np.ndarray, sims: dict, data: dict, simulate_steady_state: bool = False, 
                                     theta0_constant: np.ndarray = np.array([]), constant_indices: list = []) -> float:
    p_reconstructed = reconstruct_parameter_vector(
        p_log, theta0_constant, constant_indices)
    p = np.exp(p_reconstructed).round(ROUNDING_PRECISION)
    cost = f_cost(p, sims, data, simulate_steady_state=simulate_steady_state)

    return cost


def f_residuals(p: np.ndarray, sims: dict, data: dict, simulate_steady_state: bool = False) -> np.ndarray:
    """Return the concatenated weighted residual vector r such that sum(r**2) == f_cost.

    Every observable's (Mean - y_sim)/SEM residuals are stacked in experiment /
    observable order.  On simulation failure the residuals for the failing
    experiment are filled with 1e10 (1e10**2 == 1e20 per element, matching
    f_cost's 1e20-per-experiment penalty convention).

    A single pseudo-residual sqrt(adhoc_total) is appended so that the squared
    contribution equals the total adhoc penalty (consistent with f_cost_adhoc).
    """
    all_residuals = []
    for k_exp, d in data.items():
        observables = d["Observables"]
        n_obs_residuals = sum(len(obs["Mean"]) for obs in observables.values())
        try:
            sim = sims[k_exp]
            ic = sim.state_values
            main_time_vector = np.array(sim.time_vector, copy=True)
            sim.reset_states()
            if simulate_steady_state:
                simulate_model_steady_state(sim, ic, p)
                ic = sim.state_values

            simulate_model(sim, ic, p, time_vector=main_time_vector)

            for k_obs, obs in observables.items():
                idx = sim.feature_names.index(k_obs)
                y_sim = sim.feature_data[:, idx]
                obs_idx = np.searchsorted(sim.time_vector, obs["Time"])
                obs_idx = np.clip(obs_idx, 0, len(sim.time_vector) - 1)
                y_sim = y_sim[obs_idx]
                all_residuals.append((np.array(obs["Mean"]) - y_sim) / np.array(obs["SEM"]))

        except Exception as e:
            if "CVODE" not in str(e):
                print(f"Error in {k_exp}: {e}")
            all_residuals.append(np.full(n_obs_residuals, 1e10))

    return np.concatenate(all_residuals) if all_residuals else np.zeros(1)


def f_residuals_log_with_fixed_parameters(
    p_log: np.ndarray, sims: dict, data: dict, simulate_steady_state: bool = False,
theta0_constant: np.ndarray = np.array([]), constant_indices: list = []) -> np.ndarray:
    p_reconstructed = reconstruct_parameter_vector(p_log, theta0_constant, constant_indices)
    p = np.exp(p_reconstructed).round(ROUNDING_PRECISION)
    return f_residuals(p, sims, data, simulate_steady_state=simulate_steady_state)


def f_cost_adhoc(p: np.ndarray, sims: dict, D: dict, simulate_steady_state: bool = False, print_costs: bool = False) -> float:
    cost, adhoc = 0, 0
    for k_exp, d in D.items():
        try:
            sim = sims[k_exp]
            ic = sim.state_values
            main_time_vector = np.array(sim.time_vector, copy=True)
            sim.reset_states()
            if simulate_steady_state:
                simulate_model_steady_state(sim, ic, p)
                ic = sim.state_values

            simulate_model(sim, ic, p, time_vector=main_time_vector)

            adhoc += calculate_adhoc(sim)
            for k_obs, obs in d["Observables"].items():
                idx = sim.feature_names.index(k_obs)
                y_sim = sim.feature_data[:, idx]

                obs_idx = np.searchsorted(sim.time_vector, obs["Time"])
                obs_idx = np.clip(obs_idx, 0, len(sim.time_vector) - 1)
                y_sim = y_sim[obs_idx]
                cost += np.square((obs['Mean']-y_sim)/obs['SEM']).sum()

                if print_costs:
                    c = np.square((obs['Mean']-y_sim)/obs['SEM']).sum()
                    print(f"{k_exp}-{k_obs}: {c}")

        except Exception as e:
            if "CVODE" not in str(e):
                print(f"Error in {k_exp}: {e}")
            cost += 1e20
            return cost

    return cost + adhoc


def calculate_adhoc(sim) -> float:
    adhoc = 0

    # dummy adhoc to showcases the concept
    Rp_effect = trapezoid(sim.feature_data[:, sim.feature_names.index("Rp")], x=sim.time_vector)
    if Rp_effect < 0:
        adhoc += 1e1 + 1e0*(0 - Rp_effect)

    return adhoc
