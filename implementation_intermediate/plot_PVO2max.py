"""Plot PVo2max against exercise intensity.

This script creates a simple steady-state curve so you can quickly inspect
how PVo2max changes as intensity increases.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pvo2max_steady_state(intensity: np.ndarray, vmax: float = 80.0, km: float = 80.0, kel: float = 0.8) -> np.ndarray:
	"""Compute the steady-state PVo2max response for a given intensity."""
	intensity = np.asarray(intensity, dtype=float)
	return (vmax / kel) * intensity / (km + intensity)


def main() -> None:
	intensity = np.linspace(0.0, 100.0, 500)
	pvo2max = pvo2max_steady_state(intensity)

	figure_dir = Path(__file__).resolve().parent / "figures"
	figure_dir.mkdir(exist_ok=True)
	output_path = figure_dir / "pvo2max_vs_intensity.png"

	plt.figure(figsize=(7, 4.5))
	plt.plot(intensity, pvo2max, color="#1f77b4", linewidth=2.5)
	plt.axvline(80, color="gray", linestyle="--", linewidth=1)
	plt.text(81, pvo2max_steady_state(np.array([80.0]))[0], "80%", color="gray", va="bottom")
	plt.xlabel("Intensity (%)")
	plt.ylabel("PVo2max")
	plt.title("PVo2max vs Intensity")
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()

	print(f"Saved plot to {output_path}")


if __name__ == "__main__":
	main()
