import os

import matplotlib.pyplot as plt

from common.plotting_functions import plot_figures, plot_PI_waterfall
from common.utils import load_best_parameters
import config


def plot_results(
    model, sims: dict, estimation_data: dict, validation_data: dict,
    plot_pi: bool = False) -> None:

    # create the folder if it does not exist
    os.makedirs("./figures", exist_ok=True)
    os.makedirs(f"./figures/{model.name}", exist_ok=True)
    
    fig_names = {}

    def _infer_figure_name(fig_num: int, all_data: dict) -> str:
        fallback_title = None
        fallback_ylabel = None
        for exp_data in all_data.values():
            for obs in exp_data.get("Observables", {}).values():
                plot_info = obs.get("plotting_info", {})
                if plot_info.get("fig") != fig_num:
                    continue

                fig_name = plot_info.get("figName")
                if fig_name:
                    return fig_name

                if fallback_ylabel is None and plot_info.get("ylabel"):
                    fallback_ylabel = plot_info["ylabel"].split(" (")[0]
                if fallback_title is None and plot_info.get("title"):
                    fallback_title = plot_info["title"]

        return fallback_ylabel or fallback_title or f"Figure_{fig_num}"

    all_data = {**estimation_data, **validation_data}
    fig_numbers = sorted({
        obs.get("plotting_info", {}).get("fig")
        for exp_data in all_data.values()
        for obs in exp_data.get("Observables", {}).values()
        if obs.get("plotting_info", {}).get("fig") is not None
    })
    for fig_num in fig_numbers:
        fig_names[fig_num] = _infer_figure_name(fig_num, all_data)

    theta = load_best_parameters(
        f"./results/{model.name}", cost_key='f', model=model)

    plot_figures(
        theta, sims, all_data, model,
        fig_names,
        config.SIMULATE_STEADY_STATE)

    if plot_pi:
        plot_PI_waterfall(model.name)

    plt.close()