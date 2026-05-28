"""Plot stress against PVo2max for multiple Vmax_stress values.

This script creates steady-state curves so you can quickly inspect how stress
changes as PVo2max increases while varying Vmax_stress between 10 and 50.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def stress_steady_state(
	pvo2max: np.ndarray,
	vmax_stress: float = 10.0,
	km_stress: float = 49.0,
	n_stress: float = 1.0,
	elim_stress: float = 0.05,
) -> np.ndarray:
	"""Compute steady-state stress for a given PVo2max based on the model hill term."""
	pvo2max = np.asarray(pvo2max, dtype=float)
	hill = (vmax_stress * np.power(pvo2max, n_stress)) / (
		np.power(km_stress, n_stress) + np.power(pvo2max, n_stress)
	)
	return hill / elim_stress


def main() -> None:
	pvo2max = np.linspace(0.0, 100.0, 500)
	n_values = np.linspace(1.0, 3.0, 4)

	figure_dir = Path(__file__).resolve().parent / "figures"
	figure_dir.mkdir(exist_ok=True)
	output_path = figure_dir / "stress_vs_pvo2max_n_01_4.png"

	plt.figure(figsize=(7, 4.5))
	for n_stress in n_values:
		stress = stress_steady_state(pvo2max, n_stress=n_stress)
		plt.plot(
			pvo2max,
			stress,
			linewidth=2.0,
			label=f"N = {n_stress:.1f}",
		)
	plt.xlabel("PVo2max")
	plt.ylabel("Stress")
	plt.title("Stress vs PVo2max for varying N")
	plt.legend(fontsize=8)
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()

	print(f"Saved plot to {output_path}")


if __name__ == "__main__":
	main()
