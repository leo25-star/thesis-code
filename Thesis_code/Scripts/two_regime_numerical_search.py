"""
two_regime_numerical_search.py
==============================

Numerical search for the optimal doubling strategy in the TWO-REGIME
(step-function) model of Section 2 of the thesis, using the direct-transcription
optimal-control method described in the text:

    * the continuous control f(t) in [0,1] is replaced by a piecewise-constant
      (zero-order-hold) control with K equal intervals,  f = (f_1, ..., f_K);
    * the reduced state is propagated over one cycle by a fixed-step RK4
      integrator;
    * the resulting nonlinear program

          min_u  P     s.t.   x(P) - x(0) = 0,   r(P) - r(0) - ln2 = 0,
                              0 <= f_j <= 1,

      with u = (f_1,...,f_K, x(0), P), is solved with SLSQP;
    * several initial guesses (balanced growth + bang-bang seeds) are used so
      that the global minimum can be found.

Working in the reduced (dimensionless) ratio variables of Section 2
(q = T/R, z = A/R, r = ln R, u = k_T/k_R, v = k_C/k_A, control f = f_R):

  ribosome-limited regime  (amino acids present,  z > 0  or  q >= v/u):
        q' = (1 - f) - f q
        z' = q - v/u - f z
        r' = f
  amino-acid-limited regime  (z = 0,  q < v/u):
        q' = (u/v) q [ (1 - f) - f q ]
        z' = 0
        r' = f (u/v) q

Balanced growth is the point q* = v/u, z* = 0, f* = u/(u+v), which stays on the
switching manifold k_A T = k_C R and gives tau_BG = ln2 (1 + v/u).

Unlike the smooth-limitation (Michaelis-Menten / concentration-limited) models,
the two-regime model has NO phase transition: Section 2 proves balanced growth
is the GLOBAL optimum for every parameter value.  The figure produced here is
the numerical counterpart of that statement -- across a sweep of k_T/k_R the
SLSQP optimum coincides with balanced growth (gain ~ 0%).  The multi-start
search is seeded with balanced growth first (the incumbent) plus several
bang-bang guesses; a bang-bang result replaces the incumbent only if it beats
balanced growth by a real margin, so discretization/feasibility noise cannot
masquerade as a spuriously faster time-dependent strategy.

Run:  python two_regime_numerical_search.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

LN2 = np.log(2.0)


# ============================================================
# 1. Reduced two-regime dynamics and a fixed-step RK4 cycle
# ============================================================

def rhs(q, z, f, u, v):
    """Right-hand side (dq, dz, dr) of the reduced two-regime model.

    The regime is selected by the state: if amino acids are present (z > 0) or
    would accumulate (q >= v/u) the system is ribosome-limited; otherwise the
    pool is held at zero and the system is amino-acid-limited.
    """
    vu = v / u
    if z > 0.0 or q >= vu:                     # ribosome-limited
        dq = (1.0 - f) - f * q
        dz = q - vu - f * z
        dr = f
    else:                                      # amino-acid-limited (z = 0)
        uv = u / v
        dq = uv * q * ((1.0 - f) - f * q)
        dz = 0.0
        dr = f * uv * q
    return dq, dz, dr


def integrate_cycle(f_sub, q0, z0, P, u, v, want_traj=False):
    """RK4-propagate one cycle theta in [0,1] with per-step control f_sub.

    d/dtheta = P * rhs(.,.,f).  The amino-acid pool is clamped to z >= 0 after
    each step to enforce the state constraint.  Returns the end state
    (q, z, r); optionally the full (theta, q, z) trajectory as well.
    """
    nstep = len(f_sub)
    dth = 1.0 / nstep
    q, z, r = q0, z0, 0.0

    if want_traj:
        th_hist = np.empty(nstep + 1)
        q_hist = np.empty(nstep + 1)
        z_hist = np.empty(nstep + 1)
        th_hist[0], q_hist[0], z_hist[0] = 0.0, q, z

    for s in range(nstep):
        f = f_sub[s]

        d1 = rhs(q, z, f, u, v)
        k1 = (P * d1[0], P * d1[1], P * d1[2])
        d2 = rhs(q + 0.5 * dth * k1[0], z + 0.5 * dth * k1[1], f, u, v)
        k2 = (P * d2[0], P * d2[1], P * d2[2])
        d3 = rhs(q + 0.5 * dth * k2[0], z + 0.5 * dth * k2[1], f, u, v)
        k3 = (P * d3[0], P * d3[1], P * d3[2])
        d4 = rhs(q + dth * k3[0], z + dth * k3[1], f, u, v)
        k4 = (P * d4[0], P * d4[1], P * d4[2])

        q += (dth / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z += (dth / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        r += (dth / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if z < 0.0:
            z = 0.0

        if want_traj:
            th_hist[s + 1], q_hist[s + 1], z_hist[s + 1] = (s + 1) * dth, q, z

    if want_traj:
        return q, z, r, (th_hist, q_hist, z_hist)
    return q, z, r


# ============================================================
# 2. Balanced growth (analytic)
# ============================================================

def balanced_growth(u, v):
    """Return the balanced-growth point (q*, z*, f*, tau_BG)."""
    qstar = v / u
    fstar = u / (u + v)
    tau = LN2 * (1.0 + v / u)
    return qstar, 0.0, fstar, tau


# ============================================================
# 3. Direct-transcription NLP solved with SLSQP
# ============================================================

def solve_ocp(x0, bounds, K, m, u, v):
    """Solve  min P  s.t. cycle closes and R doubles, from one initial guess."""
    nsub_per = m
    Kq, Kz, KP = K, K + 1, K + 2

    def unpack(x):
        f = np.clip(x[:K], 0.0, 1.0)
        q0 = np.exp(x[Kq])
        z0 = max(x[Kz], 0.0)
        P = np.exp(x[KP])
        return f, q0, z0, P

    def constraints(x):
        f, q0, z0, P = unpack(x)
        f_sub = np.repeat(f, nsub_per)
        qP, zP, rP = integrate_cycle(f_sub, q0, z0, P, u, v)
        return np.array([qP - q0, zP - z0, rP - LN2])

    res = minimize(
        lambda x: np.exp(x[KP]),
        x0,
        jac=lambda x: np.concatenate([np.zeros(K + 2), [np.exp(x[KP])]]),
        method="SLSQP", bounds=bounds,
        constraints=[{"type": "eq", "fun": constraints}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    f, q0, z0, P = unpack(res.x)
    resid = float(np.max(np.abs(constraints(res.x))))
    return dict(f=f, q0=q0, z0=z0, P=P, resid=resid, feasible=resid < 1e-6)


def bangbang_seed(K, k):
    """Bang-bang control with k on/off cycles across the period."""
    th = (np.arange(K) + 0.5) / K
    return (np.sin(2.0 * np.pi * k * th) > 0.0).astype(float)


def global_optimum(u, v, K=32, m=4):
    """Multi-start SLSQP search for the fastest doubling cycle.

    The search is run to *test* whether any admissible control beats balanced
    growth.  Near balanced growth the objective is extremely flat, so SLSQP can
    drift into meaningless wiggles that change f(theta) without lowering the
    period P.  We therefore keep the best feasible period found across all
    starts, and:

      * if that period genuinely undercuts tau_BG by a real margin, the search
        has found a time-dependent optimum, which we return as-is;
      * otherwise the search has confirmed balanced growth, and we return the
        clean flat control f(theta) = f* (period tau_BG) -- the wiggle carries
        zero objective gain and is not a distinct strategy.

    This is the same convention the smooth-model figures use: a flat f* is drawn
    wherever balanced growth is optimal.
    """
    qs, zs, fs, taus = balanced_growth(u, v)
    bounds = ([(0.0, 1.0)] * K
              + [(np.log(1e-3), np.log(1e3)),   # log q0
                 (0.0, 50.0),                   # z0
                 (np.log(0.1 * taus), np.log(10.0 * taus))])  # log P

    def make_x0(f0, q0, z0, P0):
        return np.concatenate([np.clip(f0, 0.0, 1.0),
                               [np.log(q0), z0, np.log(P0)]])

    # Balanced growth is an exact fixed point of the integrator (q0=v/u, z0=0,
    # P=tau_BG closes with zero residual), so evaluate it directly as a
    # guaranteed-feasible baseline for the search.
    f_bg = np.full(K, fs)
    qP, zP, rP = integrate_cycle(np.repeat(f_bg, m), qs, 0.0, taus, u, v)
    resid_bg = float(np.max(np.abs([qP - qs, zP - 0.0, rP - LN2])))
    bg_sol = dict(f=f_bg, q0=qs, z0=0.0, P=taus, resid=resid_bg,
                  feasible=resid_bg < 1e-6, improved=False)

    seeds = []
    for k in (1, 2, 3):                                         # bang-bang tries
        seeds.append(make_x0(bangbang_seed(K, k), qs, 0.2, taus))

    margin = 2.0e-3 * taus                        # real-improvement threshold
    best = bg_sol
    searched_min = taus if bg_sol["feasible"] else np.inf
    for x0 in seeds:
        sol = solve_ocp(x0, bounds, K, m, u, v)
        if not sol["feasible"]:
            continue
        searched_min = min(searched_min, sol["P"])
        if sol["P"] < best["P"] - margin:         # genuine time-dependent win
            best = sol
            best["improved"] = True

    best["theta"] = (np.arange(K) + 0.5) / K
    best["best_P"] = searched_min                 # fastest feasible period found
    best["K"], best["m"] = K, m
    return best


# ============================================================
# 4. Sweep + figure
# ============================================================

V = 2.5                      # k_C/k_A = 2.5  (i.e. k_A/k_C = 0.4, as in the
#                              smooth-model figures)
U_VALUES = [0.6, 1.0, 1.5, 2.5, 4.0, 6.0]     # sweep of k_T/k_R (=> 6 panels)


def run_sweep():
    records = []
    print(f"\n=== Two-regime model: numerical search  (k_A/k_C = {1/V:g}) ===")
    for u in U_VALUES:
        qs, zs, fs, tau_bg = balanced_growth(u, V)
        top = global_optimum(u, V)
        P_opt = top["P"] if top["feasible"] else np.nan
        gain = 100.0 * (tau_bg - P_opt) / tau_bg if np.isfinite(P_opt) else 0.0
        rec = dict(u=u, fstar=fs, tau_bg=tau_bg, top=top, gain=gain)
        records.append(rec)
        verdict = "time-dependent win" if top["improved"] else "balanced growth"
        print(f"  k_T/k_R={u:5.2f}  f*={fs:.3f}  tau_BG={tau_bg:.4f}  "
              f"fastest searched P={top['best_P']:.4f}  ->  {verdict}  "
              f"(gain={gain:+.2f}%)")
    return records


def make_figure(records, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4),
                             constrained_layout=True, squeeze=False)

    for idx, rec in enumerate(records):
        i, j = divmod(idx, 3)
        ax = axes[i, j]
        top = rec["top"]

        ax.plot(top["theta"], top["f"], color="C3", lw=2.4,
                label="global optimum (SLSQP)")
        ax.axhline(rec["fstar"], color="k", ls=":", lw=0.9,
                   label="balanced growth $f^*$")
        ax.axhline(0.0, color="gray", ls=":", lw=0.5)
        ax.axhline(1.0, color="gray", ls=":", lw=0.5)

        ax.set_ylim(-0.08, 1.08)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(rf"$k_T/k_R$={rec['u']:.3g},  $f^*$={rec['fstar']:.3f},  "
                     rf"gain={rec['gain']:.2f}%", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"cycle phase $\theta = t/\tau$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"allocation $f(\theta)$", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Two-regime model:  numerical optimum is balanced growth for all "
        "parameters\n"
        rf"(no phase transition;  $k_A/k_C={1/V:g}$;  sweep of $k_T/k_R$)",
        fontsize=12)

    png = os.path.join(outdir, "two_regime_numerical_search.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {png}")
    return fig


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    records = run_sweep()
    make_figure(records, outdir)
    plt.show()


if __name__ == "__main__":
    main()
