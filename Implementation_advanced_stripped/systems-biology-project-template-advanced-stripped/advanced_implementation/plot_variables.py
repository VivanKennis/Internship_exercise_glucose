from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sund

from common.handle_data import load_and_process_data
from common.simulation import create_sims_from_data, simulate_model
from common.utils import load_best_parameters

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "M_epinephrine18"
MODEL_FILE = BASE_DIR / "models" / f"{MODEL_NAME}.txt"
DATA_FILE = BASE_DIR / "data" / "data_epinephrine.json"
RESULTS_DIR = BASE_DIR / "results" / MODEL_NAME
FIGURES_DIR = BASE_DIR / "figures" / "variables"

import sund
try:
    sund.uninstall_model(MODEL_NAME)
except:
    pass
sund.install_model(f"./models/{MODEL_NAME}.txt")

MODEL_VARIABLES = [
	"PVo2max",
	"lactate",
	"O2_deficit",
	"met_stress",
	"exercise_drive",
    "NOR_neuronal",
	"NOR_adrenal",
    "Epi_adrenal",
	"EPI",
	"NOR"
]

EXPECTED_MODEL_FEATURES = {"NOR_neuronal", "NOR_adrenal", "Epi_adrenal"}

DATA_ALIASES = {
	"EPI": "Epinephrine_nmolL",
	"NOR": "Norepinephrine_nmolL",
	"lactate": "Lactate_mmolL"
}

DISPLAY_LABELS = {
	"PVo2max": "PVo2max",
	"lactate": "Lactate",
	"O2_deficit": "Oxygen",
	"met_stress": "metabolic stress",
	"NOR_neuronal": "Noradrenaline nerves",
	"NOR_adrenal": "Noradrenaline adrenal glands",
	"Epi_adrenal": "Epinephrine adrenal glands",
	"EPI": "Epinephrine",
	"NOR": "Norepinephrine",
	"exercise_drive": "Exercise Drive"
}

def sanitize_filename(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def build_time_vector(all_times: np.ndarray) -> np.ndarray:
	t_end = float(np.max(all_times))
	if t_end <= 0:
		return np.array([0.0])
	step = 0.1
	n_points = int(math.floor(t_end / step)) + 1
	return np.linspace(0.0, t_end, max(n_points, 2))


def get_feature_series(sim, variable_name: str):
	"""Get a time series for a variable from feature/state trajectories."""
	def _extract_series(values, idx: int):
		if values is None:
			return None
		if callable(values):
			values = values()
		if not hasattr(values, "ndim"):
			values = np.asarray(values)
		if values.ndim == 2:
			return values[:, idx]
		if values.ndim == 1:
			# Only accept 1D arrays that already match the time vector length.
			if len(values) == len(sim.time_vector):
				return values
		return None

	series_sources = [
		("feature_names", ["feature_values"]),
		("state_names", ["state_values"]),
	]
	for names_attr, value_attrs in series_sources:
		names = getattr(sim, names_attr, None)
		if names is None or variable_name not in names:
			continue
		idx = names.index(variable_name)
		for value_attr in value_attrs:
			series = _extract_series(getattr(sim, value_attr, None), idx)
			if series is not None:
				return series
	return None

def get_observable_for_variable(observables: dict, variable_name: str):
	observable_name = DATA_ALIASES.get(variable_name, variable_name)
	return observables.get(observable_name)


def add_intensity_markers(ax, experiment_data: dict):
	intensity = experiment_data.get("input", {}).get("intensity")
	if not intensity or intensity.get("type", "").lower() != "piecewise_constant":
		return

	times = intensity.get("t", [])
	values = intensity.get("f", [])
	for index in range(1, len(times)):
		try:
			time_value = float(times[index])
		except (TypeError, ValueError):
			continue
		if not np.isfinite(time_value) or time_value <= 0:
			continue

		ax.axvline(time_value, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
		if index < len(values):
			ax.text(
				time_value,
				0.96,
				f"{values[index]}%",
				transform=ax.get_xaxis_transform(),
				ha="center",
				va="top",
				fontsize=7,
				color="gray",
				bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
			)


def plot_experiment(experiment_name: str, experiment_data: dict, sim, params) -> None:
	sim.reset_states()
	simulate_model(sim, sim.state_values, params, time_vector=build_time_vector(experiment_data["all_times"]))

	n_vars = len(MODEL_VARIABLES)
	n_cols = 3
	n_rows = math.ceil(n_vars / n_cols)

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.1 * n_rows), squeeze=False)
	fig.suptitle(experiment_name, fontsize=14, fontweight="bold")

	for index, variable_name in enumerate(MODEL_VARIABLES):
		ax = axes[index // n_cols][index % n_cols]
		y_sim = get_feature_series(sim, variable_name)
		if y_sim is not None:
			ax.plot(sim.time_vector, y_sim, color="#ff7f0e", linewidth=2, label="Model")
		else:
			ax.text(0.5, 0.5, "No simulation output", transform=ax.transAxes, ha="center", va="center", fontsize=9)

		observable = get_observable_for_variable(experiment_data["Observables"], variable_name)
		if observable is not None:
			ax.plot(
				observable["Time"],
				observable["Mean"],
				color= "#1f77b4",
				marker="o",
				markersize=4,
				markerfacecolor="none",
				linewidth=1.8,
				label="Data",
			)
			ax.fill_between(
				observable["Time"],
				observable["Mean"] - observable["SEM"],
				observable["Mean"] + observable["SEM"],
				color="#1f77b4",
				alpha=0.18,
			)

		add_intensity_markers(ax, experiment_data)
		ax.set_title(DISPLAY_LABELS.get(variable_name, variable_name), fontsize=10)
		ax.set_xlabel("Time (minutes)")
		ax.grid(alpha=0.2)

		if variable_name in {"Glucose_mmolL", "G", "Glucose_mgdl"}:
			ax.set_ylabel("Glucose (mmol/L)")
		elif variable_name in {"I", "Insulin_mUL"}:
			ax.set_ylabel("Insulin (mU/L)")
		else:
			ax.set_ylabel(variable_name)

		if observable is not None or y_sim is not None:
			ax.legend(fontsize=8, loc="best")

	for index in range(n_vars, n_rows * n_cols):
		fig.delaxes(axes[index // n_cols][index % n_cols])

	fig.tight_layout(rect=(0, 0, 1, 0.97))
	output_name = sanitize_filename(experiment_name)
	output_path = FIGURES_DIR / f"{output_name}_all_variables.png"
	fig.savefig(output_path, dpi=250)
	plt.close(fig)
	print(f"Saved {output_path}")


def main() -> None:
	os.chdir(BASE_DIR)
	FIGURES_DIR.mkdir(exist_ok=True)

	try:
		model = sund.load_model(MODEL_NAME)
	except Exception:
		print("Compiled model not available yet, installing model first...")
		sund.install_model(f"./models/{MODEL_NAME}.txt")
		model = sund.load_model(MODEL_NAME)

	model_feature_names = set(getattr(model, "feature_names", []) or [])
	if not EXPECTED_MODEL_FEATURES.issubset(model_feature_names):
		missing = sorted(EXPECTED_MODEL_FEATURES - model_feature_names)
		print(f"Model build is missing expected features {missing}; reinstalling model from source...")
		sund.install_model(f"./models/{MODEL_NAME}.txt")
		model = sund.load_model(MODEL_NAME)

	data = load_and_process_data(str(DATA_FILE))
	sims = create_sims_from_data(model, data)
	params = load_best_parameters(str(RESULTS_DIR), cost_key='f', model=model)

	for experiment_name, experiment_data in data.items():
		plot_experiment(experiment_name, experiment_data, sims[experiment_name], params)


if __name__ == "__main__":
    main()
