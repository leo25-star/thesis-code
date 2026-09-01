"""
phase_transition_summary_plots.py
=================================

One summary figure collecting the balanced-growth (BG) phase transition for the
four models studied in the thesis.  For each model the plane is split into

  * a BG-optimal region      (constant allocation is locally optimal), and
  * a time-dependent region  (BG loses local optimality; a cyclic / bang-bang
                              strategy does better),

separated by the analytically derived transition boundary.

Throughout, the transporter-to-ribosome rate ratio is
        u = k_T / k_R ,
K_m is the Michaelis--Menten saturation constant, and chi is the relative
transporter weight in the effective volume V = R + chi*T.

The four models and their boundaries
------------------------------------
  (a) Two-regime (step-function) model.
        Balanced growth is ALWAYS optimal -- there is no transition; the whole
        parameter plane is a single BG phase.

  (b) Michaelis--Menten with R in the denominator,
        S(q,z) = z / ( z + 1 + K_m (1 + chi q) ).
        BG optimality is set by the sign of the second-variation coefficient
        M_chi, which vanishes on
                K_m = 1 / (chi u - 1)          (only meaningful for chi u > 1).
        BG is optimal where  K_m (chi u - 1) < 1  (i.e. M_chi > 0).

  (c) Michaelis--Menten with the classic factor (no R in the denominator),
        S(q,z) = z / (z + 1 + K_m).
        The transition collapses onto
                u = 1        (k_T = k_R),
        independent of K_m.  BG is optimal for u < 1.

  (d) Large-K_m limit -- the concentration-limited (A/V scaling) model,
        rho(q,z) = z / (1 + chi q).
        Obtained as the K_m -> infinity limit of the MM model; the transition
        again occurs at
                u = 1        (k_T = k_R).

Output: ``phase_transition_summary.png`` (written next to this script).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Relative transporter weight in V = R + chi*T.  With chi = 1 the finite-K_m
# boundary of model (b), K_m = 1/(chi u - 1), asymptotes to u = 1 as K_m -> inf,
# smoothly connecting to the large-K_m models (c) and (d).
CHI = 1.0

# Plotting window.
U_MIN, U_MAX = 0.2, 4.0
KM_MIN, KM_MAX = 1e-2, 1e2

# Two-phase colour scheme (shared across every panel).
C_BG = "#9ecae1"        # BG optimal (constant allocation)
C_DYN = "#fdae6b"       # time-dependent optimal
PHASE_CMAP = ListedColormap([C_BG, C_DYN])


def _grid(n_u=400, n_km=400):
    """(u, K_m) mesh; K_m logarithmic."""
    u = np.linspace(U_MIN, U_MAX, n_u)
    km = np.logspace(np.log10(KM_MIN), np.log10(KM_MAX), n_km)
    return np.meshgrid(u, km)          # U, KM


def _shade(ax, U, KM, dynamic):
    """Fill the plane: BG-optimal vs time-dependent, from a boolean `dynamic`."""
    ax.pcolormesh(U, KM, dynamic.astype(float), cmap=PHASE_CMAP,
                  vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set_xlim(U_MIN, U_MAX)
    ax.set_ylim(KM_MIN, KM_MAX)
    ax.set_xlabel(r"transporter/ribosome rate ratio  $u=k_T/k_R$")
    ax.set_ylabel(r"saturation constant  $K_m$")


# ============================================================
# One panel per model
# ============================================================

def panel_step(ax):
    """(a) Two-regime step-function model: BG optimal everywhere."""
    U, KM = _grid()
    dynamic = np.zeros_like(U, dtype=bool)          # never time-dependent
    _shade(ax, U, KM, dynamic)
    ax.text(0.5, 0.5, "BG optimal\neverywhere\n(no transition)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#08519c")
    ax.set_title("(a) Two-regime (step-function) model", fontsize=11)


def panel_mm_R(ax):
    """(b) MM with R in the denominator: K_m = 1/(chi u - 1)."""
    U, KM = _grid()
    # M_chi > 0  <=>  K_m (chi u - 1) < 1  ->  BG optimal.
    dynamic = KM * (CHI * U - 1.0) > 1.0
    _shade(ax, U, KM, dynamic)

    # Transition curve K_m = 1/(chi u - 1), drawn only where chi u > 1.
    u_line = np.linspace(1.0 / CHI + 1e-3, U_MAX, 400)
    km_line = 1.0 / (CHI * u_line - 1.0)
    inwin = (km_line >= KM_MIN) & (km_line <= KM_MAX)
    ax.plot(u_line[inwin], km_line[inwin], "k-", lw=2.2)
    # Vertical asymptote u = 1/chi (the K_m -> inf limit of this boundary).
    ax.axvline(1.0 / CHI, color="k", ls=":", lw=1.2)
    ax.text(1.0 / CHI, KM_MAX * 0.55, r"$u=1/\chi$", rotation=90,
            ha="right", va="top", fontsize=9)

    ax.set_title(r"(b) MM with $R$ in denominator:  "
                 r"$K_m=\dfrac{1}{\chi u-1}$" + f"   ($\\chi={CHI:g}$)",
                 fontsize=11)


def panel_mm_classic(ax):
    """(c) MM classic factor (no R in the denominator): u = 1."""
    U, KM = _grid()
    dynamic = U > 1.0
    _shade(ax, U, KM, dynamic)
    ax.axvline(1.0, color="k", lw=2.2)
    ax.set_title(r"(c) MM classic factor (no $R$):  $u=1$", fontsize=11)


def panel_large_km(ax):
    """(d) Large-K_m / concentration (A/V) model: u = 1."""
    U, KM = _grid()
    dynamic = U > 1.0
    _shade(ax, U, KM, dynamic)
    ax.axvline(1.0, color="k", lw=2.2)
    # Emphasise that K_m plays no role here (this is the K_m -> inf limit).
    ax.text(0.03, 0.97, r"$K_m\to\infty$ limit"
            "\n" r"($A/V$ scaling)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.set_title(r"(d) Large-$K_m$ (concentration $A/V$) model:  $u=1$",
                 fontsize=11)


# ============================================================
# Figure
# ============================================================

def make_figure(outdir):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5),
                             constrained_layout=True)
    panel_step(axes[0, 0])
    panel_mm_R(axes[0, 1])
    panel_mm_classic(axes[1, 0])
    panel_large_km(axes[1, 1])

    legend_handles = [
        Patch(facecolor=C_BG, edgecolor="none",
              label="BG optimal (constant allocation)"),
        Patch(facecolor=C_DYN, edgecolor="none",
              label="time-dependent optimal (BG unstable)"),
        Line2D([0], [0], color="k", lw=2.2, label="transition boundary"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Balanced-growth phase transition across the models",
                 fontsize=14, fontweight="bold")

    png = os.path.join(outdir, "phase_transition_summary.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {png}")
    return fig


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    make_figure(outdir)
    plt.show()


if __name__ == "__main__":
    main()
