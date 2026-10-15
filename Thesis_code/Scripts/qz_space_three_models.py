"""
qz_space_three_models.py
========================

Companion figure to `phase_transition_three_models.py`.

That script draws, for each of the three smooth-limitation models of Section 3,
the *optimal allocation* f(theta) on a 2x3 grid -- three parameter values before
the phase transition (M>0, bang-bang gives nothing, BG is optimal) and three
after it (M<0, the optimum oscillates).

Here we draw the SAME 2x3 grid, but instead of the control we show the resulting
*dynamics in the (q, z) plane*, i.e. the closed orbit traced by

    q = T/R        (transporter-to-ribosome ratio)
    z = A/R        (precursor-to-ribosome ratio)

over one cell cycle theta = t/tau in [0, 1], obtained by integrating

    dq/dtheta = P * phi(q,z) [ (1 - f(theta)) - f(theta) q ]
    dz/dtheta = P * [ q - phi(q,z) ( v/u + f(theta) z ) ]

with the optimal piecewise-constant f(theta) and the optimal period P found by
`phase_transition_three_models.analyse_point`.

Before the transition the optimum is balanced growth, so the "orbit" collapses
onto the BG fixed point (q*, z*) and is drawn as a single marker.  After the
transition it opens into a closed loop -- that opening *is* the phase
transition, seen in state space rather than in the control.

Each panel also shows the two nullclines of the frozen-control system evaluated
at the BG allocation f = f*,

    dq/dtheta = 0   ->   q = (1 - f*)/f*      (vertical line)
    dz/dtheta = 0   ->   q = phi(q,z) (v/u + f* z),

whose intersection is the BG point; they give the orbit a fixed reference frame.

Usage
-----
    python qz_space_three_models.py            # full search + (q,z) figures
    python qz_space_three_models.py MM         # one model only

`phase_transition_three_models.py` calls `fig_qz_space` automatically after it
has drawn its own strategy figure, so running that script produces both sets.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import phase_transition_three_models as pt3

# single colour for the optimal orbit
ORBIT_COLOR = "blue"


# ============================================================
# 1. Integrate one cycle with the optimal piecewise-constant f
# ============================================================

def integrate_qz(model, P, q0, z0, Pd, f_vals, n_sub=16):
    """Integrate (q, z) over theta in [0,1] with f piecewise constant.

    `f_vals[i]` is the control on the i-th of the K = len(f_vals) equal
    sub-intervals of the cycle, exactly as returned by the optimal-control
    search.  Returns (theta, q, z) sampled densely inside every sub-interval.
    """
    K = len(f_vals)

    def rhs(_th, Y, f):
        q, z = Y
        return [Pd * pt3.F_fun(model, q, z, f, P),
                Pd * pt3.G_fun(model, q, z, f, P)]

    th_all = [0.0]
    q_all = [q0]
    z_all = [z0]
    Y = np.array([q0, z0], dtype=float)

    for i in range(K):
        t0, t1 = i / K, (i + 1) / K
        t_eval = np.linspace(t0, t1, n_sub + 1)[1:]
        sol = solve_ivp(rhs, (t0, t1), Y, args=(float(f_vals[i]),),
                        method="LSODA", rtol=1e-10, atol=1e-12, t_eval=t_eval)
        if not sol.success or sol.y.shape[1] == 0:
            break
        th_all.extend(sol.t)
        q_all.extend(sol.y[0])
        z_all.extend(sol.y[1])
        Y = sol.y[:, -1]

    return (np.asarray(th_all), np.asarray(q_all), np.asarray(z_all))


def orbit_of_record(model, rec, n_sub=16):
    """(theta, q, z) of the optimal cycle stored in one sweep record."""
    return integrate_qz(model, rec["Pvec"], rec["q0"], rec["z0"],
                        rec["P_cycle"], rec["f_opt"], n_sub=n_sub)


# ============================================================
# 2. Panel drawing helpers
# ============================================================

def _draw_nullclines(ax, model, P, f_star, qlim, zlim, ngrid=241):
    """Nullclines of the frozen system at the BG allocation f = f*."""
    qq = np.linspace(qlim[0], qlim[1], ngrid)
    zz = np.linspace(zlim[0], zlim[1], ngrid)
    Q, Z = np.meshgrid(qq, zz)

    Fv = pt3.F_fun(model, Q, Z, f_star, P)
    Gv = pt3.G_fun(model, Q, Z, f_star, P)

    ax.contour(Q, Z, Fv, levels=[0.0], colors="0.55",
               linestyles="--", linewidths=0.9)
    ax.contour(Q, Z, Gv, levels=[0.0], colors="0.55",
               linestyles="-.", linewidths=0.9)


def _draw_orbit(ax, theta, q, z):
    """Closed orbit as a single-colour line, with flow arrows."""
    line, = ax.plot(q, z, color=ORBIT_COLOR, lw=2.2)

    # a few direction arrows along the orbit
    idx = np.linspace(0, len(q) - 2, 7).astype(int)[:-1]
    for i in idx:
        ax.annotate("", xy=(q[i + 1], z[i + 1]), xytext=(q[i], z[i]),
                    arrowprops=dict(arrowstyle="-|>", color="0.25",
                                    lw=0.0, mutation_scale=13))
    return line


def _panel(ax, model, rec, show_ylabel=False):
    P, f_star = rec["Pvec"], rec["f"]
    q_bg, z_bg = rec["q_bg"], rec["z_bg"]
    theta, q, z = orbit_of_record(model, rec)

    closed = bool(len(q) > 1 and np.isfinite(q[-1]) and np.isfinite(z[-1]))
    amp = (max(np.ptp(q), 1e-12) / max(q_bg, 1e-12) if closed else 0.0)

    if closed and len(q) > 2 and np.ptp(q) > 1e-9 * q_bg:
        qlo, qhi = np.min(q), np.max(q)
        zlo, zhi = np.min(z), np.max(z)
        padq = 0.28 * max(qhi - qlo, 0.05 * q_bg)
        padz = 0.28 * max(zhi - zlo, 0.05 * z_bg)
    else:                                   # BG optimum: orbit is a point
        qlo = qhi = q_bg
        zlo = zhi = z_bg
        padq, padz = 0.16 * q_bg, 0.16 * z_bg

    qlim = (max(1e-6, qlo - padq), qhi + padq)
    zlim = (max(1e-6, zlo - padz), zhi + padz)

    if closed and np.ptp(q) > 1e-9 * q_bg:
        _draw_orbit(ax, theta, q, z)
    ax.plot(q_bg, z_bg, "*", ms=14, color="k", zorder=6,
            label="BG fixed point")

    ax.set_xlim(*qlim)
    ax.set_ylim(*zlim)
    ax.grid(True, lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    if show_ylabel:
        ax.set_ylabel(r"$z$", fontsize=16)

    return amp


# ============================================================
# 3. The 2x3 figure
# ============================================================

def split_records(records):
    """Three records just before the transition and three just after."""
    recs = sorted(records, key=lambda r: r["M"], reverse=True)
    before = sorted([r for r in recs if r["M"] > 0.0][-3:],
                    key=lambda r: r["sweep_val"])
    after = sorted([r for r in recs if r["M"] <= 0.0][:3],
                   key=lambda r: r["sweep_val"])
    return before, after


def fig_qz_space(model, cfg, P, star, records, outdir):
    """2x3 grid of the optimal (q, z) orbits, before / after the transition."""
    key = cfg["sweep"]
    before, after = split_records(records)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4),
                             constrained_layout=True, squeeze=False)
    rows = [before, after]
    row_label = ["before transition\n(BG optimal, $M>0$)",
                 "after transition\n(BG not optimal, $M<0$)"]
    sweep_sym = r"u" if key == "u" else r"K_m"

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            if j >= len(rows[i]):
                ax.axis("off")
                continue
            r = rows[i][j]
            amp = _panel(ax, model, r, show_ylabel=(j == 0))
            extra = ",  BG point" if amp < 1e-6 else ""
            ax.set_title(rf"${sweep_sym}$={r['sweep_val']:.3g}{extra}",
                         fontsize=14)

    for i in range(2):
        axes[i, 0].annotate(row_label[i], xy=(0, 0.5), xytext=(-0.52, 0.5),
                            xycoords="axes fraction", textcoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=13, fontweight="bold")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$q$", fontsize=16)

    handles = [
        plt.Line2D([], [], color=ORBIT_COLOR, lw=2.2,
                   label="optimal orbit found by SLSQP"),
        plt.Line2D([], [], marker="*", ls="none", color="k", ms=14,
                   label="BG fixed point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, -0.06))

    if key == "u":
        trans = rf"transition at $u=1/\chi={star:.3g}$"
        fixed = rf"$v={P[pt3.IV]:g}$, $\chi={P[pt3.ICHI]:g}$"
        if model != "conc":   # phi of the concentration-limited model has no K_m
            fixed += rf", $K_m={P[pt3.IKM]:g}$"
    else:
        trans = rf"transition at $K_m={star:.3g}$"
        fixed = rf"$v={P[pt3.IV]:g}$, $\chi={P[pt3.ICHI]:g}$, $u={P[pt3.IU]:g}$"
    fig.suptitle(f"{cfg['title']}:  optimal dynamics in the $(q,z)$ plane\n"
                 f"({trans};  {fixed})", fontsize=17)

    png = os.path.join(outdir, f"qz_space_{cfg['tag']}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {png}")
    return fig


# ============================================================
# 4. Stand-alone driver (same sweep as phase_transition_three_models)
# ============================================================

def run_model_qz(name, cfg, outdir):
    key = cfg["sweep"]
    start = np.sqrt(cfg["lo"] * cfg["hi"])
    P = pt3.set_param(cfg["P"], key, start)

    print(f"\n=== {cfg['title']}  (sweep {key}) ===")
    star, _rows = pt3.find_transition(name, P, key, cfg["lo"], cfg["hi"])
    if star is None:
        print("  No M sign-change found in the scanned range.")
        return None

    pred = (1.0 / P[pt3.ICHI] if key == "u"
            else 1.0 / (P[pt3.ICHI] * P[pt3.IU] - 1.0))
    print(f"  transition (numeric): {key}* = {star:.4f}   "
          f"(analytic prediction {pred:.4f})")

    records = pt3.build_sweep(name, P, key, star)
    if len(records) < 2:
        print("  Too few valid sweep points for a figure.")
        return star
    fig_qz_space(name, cfg, P, star, records, outdir)
    return star


def main():
    # Figures are written to Thesis_code/PNGs (one level up from Scripts/)
    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PNGs"
    )
    os.makedirs(outdir, exist_ok=True)
    wanted = [a for a in sys.argv[1:] if a in pt3.MODELS]
    names = wanted if wanted else list(pt3.MODELS)
    for name in names:
        run_model_qz(name, pt3.MODELS[name], outdir)
    plt.show()


if __name__ == "__main__":
    main()
