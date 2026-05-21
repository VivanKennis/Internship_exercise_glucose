import os
import numpy as np
import matplotlib.pyplot as plt


def pvo2max_ss(u3, Vmax=80.0, Km=80.0, Kel=0.8):
    """Steady-state PVo2max for Michaelis-Menten input-response.

    P_ss = (Vmax/Kel) * u3/(Km + u3)
    u3 and Km are in percent (0-100), PVo2max is percent-equivalent (dimensieloos).
    """
    u3 = np.asarray(u3, dtype=float)
    denom = Km + u3
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    return (Vmax / Kel) * (u3 / denom)


def make_plot(out_path="figures/pvo2max.png"):
    u3 = np.linspace(0, 100, 501)
    Vmax = 80.0
    Km = 80.0
    Kel = 0.8
    P = pvo2max_ss(u3, Vmax=Vmax, Km=Km, Kel=Kel)

    out_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(out_path)) if not os.path.isabs(out_path) else out_path

    plt.figure(figsize=(6,4))
    plt.plot(u3, P, color="#1f77b4", lw=2)
    plt.fill_between(u3, P, alpha=0.12, color="#1f77b4")
    plt.axvline(80, color="gray", ls="--", lw=1)
    plt.text(82, np.interp(82, u3, P), "80% threshold", va="center", color="gray")
    plt.xlabel("Intensity (%)")
    plt.ylabel("PVo2max (percent-equivalent)")
    plt.title("Steady-state PVo2max vs Intensity (MM form)")
    plt.grid(alpha=0.25)
    plt.xlim(0,100)
    plt.ylim(0, max(120, P.max()*1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pvo2max.png"), dpi=200)
    print(f"Saved PVo2max plot to {os.path.join(out_dir, 'pvo2max.png')}")


if __name__ == '__main__':
    make_plot()
