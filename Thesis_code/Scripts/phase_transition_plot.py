"""
Phase_transition_optimal.py
===========================

Picture of the optimal strategy as the balanced-growth (BG) point loses its
optimality, for the TWO distinct ways it can happen.

Background (continues Declining_direction_search / Negative_direction_check)
---------------------------------------------------------------------------
The BG strategy is a singular arc of the doubling-time problem.  Its local
optimality is governed by the reduced second variation

        delta^2 tau[y_n] = M I_ydot(n)/tau + N tau I_y(n),
        y_n(theta) = sin^2(pi theta) sin(n pi theta),  theta = t/tau ,

with I_ydot(n) ~ n^2 and I_y(n) = O(1).  BG is optimal iff delta^2 tau[y_n] >= 0
for every mode n.  There are two boundaries where this fails:

  (A) M = 0  with  N > 0   -- HIGH-FREQUENCY / singular-arc transition.
        For M < 0 the form is unbounded below; the smallest declining mode is
              n_min ~ sqrt(-N / M)  ->  infinity as M -> 0^- .
        M is the Legendre--Clebsch (Kelley) coefficient of the singular arc;
        M < 0 violates it, so the optimum LEAVES the arc and is bang-bang.
        The critical mode runs off to infinite frequency, so no single smooth
        mode generates the optimum: the declining direction connects to it only
        through its switching FREQUENCY (n_min ~ number of switches), not shape.

  (B) criterion = N + (pi^2/tau^2) M = 0  with  M > 0  -- LOW-FREQUENCY
        transition.  Here the critical mode is finite (n = 1): a soft
        bifurcation in which the optimum emerges continuously ALONG the
        declining direction.  Here the declining direction does predict the
        optimal shape.

This script locates both boundaries (by sweeping k_T), and at each point builds
and graphs:

  * declining direction   -- linear mode n_crit, control eta_n(t).
  * ray minimum           -- best doubling time on f = f* + alpha*eta_n.
  * global optimum        -- genuine optimal control: minimise doubling time
                             over ALL admissible f(t) in [0,1] (direct
                             piecewise-constant transcription).
  * fundamental mode      -- the dominant Fourier harmonic of the global
                             optimum, to test the frequency connection.

Figures written (one set per boundary, tag = Mzero / criterion):
  * phase_transition_strategies_<tag>.png  -- per-M panel: the four curves.
  * phase_transition_scaling_<tag>.png     -- order parameters vs distance.
  * phase_transition_frequency_<tag>.png   -- switches vs n_crit (the link).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Second_variation_calculation import (  # noqa: E402
    F_fun, G_fun, lam_fun, compute_MN, first_order_coeffs,
)
from Declining_direction_search import (  # noqa: E402
    eta_time, second_variation, solve_perturbed_doubling,
)

LN2 = np.log(2.0)

KIND_TAG = {"M": "Mzero", "criterion": "criterion"}
KIND_LABEL = {
    "M": "M = 0  (high-frequency / singular-arc transition)",
    "criterion": "criterion = 0  (low-frequency, mode 1 transition)",
}


# ============================================================
# 0. Second-variation helpers
# ============================================================

def critical_mode(M, N, tau, nmax=80):
    """Smallest mode n with delta^2 tau[y_n] < 0, or None if BG is optimal.

    Handles both mechanisms uniformly:
      M < 0            -> finite n_min ~ sqrt(-N/M)  (high-frequency boundary)
      M > 0, N < 0     -> n = 1 typically            (low-frequency boundary)
      M >= 0, N >= 0   -> None  (BG is the optimum)
    """
    if M >= 0.0 and N >= 0.0:
        return None
    for n in range(1, nmax + 1):
        if second_variation(M, N, tau, n) < 0.0:
            return n
    return None


def dominant_harmonic(f, theta):
    """Fit the single strongest Fourier harmonic of f(theta) over one cycle.

    Returns (component, k, amp): the reconstructed harmonic mean + a*cos + b*sin
    on the same theta grid, its harmonic index k (cycles per period), and its
    amplitude.  A clean k-harmonic square wave has 2k switches, so 2k is the
    optimum's switch frequency.
    """
    f = np.asarray(f, float)
    fc = f - f.mean()
    if np.allclose(fc, 0.0):
        return np.full_like(f, f.mean()), 0, 0.0
    mag = np.abs(np.fft.rfft(fc))
    k = int(np.argmax(mag[1:]) + 1)                 # skip DC
    c = np.cos(2 * np.pi * k * theta)
    s = np.sin(2 * np.pi * k * theta)
    A = np.vstack([c, s]).T
    coef, *_ = np.linalg.lstsq(A, fc, rcond=None)
    comp = f.mean() + A @ coef
    return comp, k, float(np.hypot(*coef))


# ============================================================
# 1. Locate a transition in k_T (generic over the order parameter)
# ============================================================

def mn_at(kT, kA, Km, guess, kR=1.0, kC=1.0):
    """compute_MN wrapper returning a tidy record (or None)."""
    out = compute_MN(kR, kT, kA, kC, Km, guess=guess)
    if out is None:
        return None
    out["kT"], out["kA"], out["Km"], out["kR"], out["kC"] = kT, kA, Km, kR, kC
    return out


def coarse_scan(kA, Km, kT_grid):
    """Scan k_T and keep well-behaved BG points (continuation in guess)."""
    rows = []
    guess = (1.0, 1.0)
    for kT in kT_grid:
        out = mn_at(kT, kA, Km, guess)
        if out is None:
            continue
        guess = out["guess"]
        if not (np.isfinite(out["M"]) and np.isfinite(out["N"])
                and np.isfinite(out["criterion"])):
            continue
        if not (0.05 < out["f"] < 0.95):
            continue
        if out["lambda"] <= 0.05 or abs(out["Hf"]) > 1e-2:
            continue
        rows.append(out)
    return rows


def find_transition(kA, Km, key="M", kT_lo=1.2, kT_hi=200.0, n_coarse=26,
                    require_Mpos=False):
    """Bracket and bisect the k_T where `key` changes sign (stable -> unstable).

    Convention: stable side has key > 0, unstable side key < 0.
    require_Mpos keeps the bracket inside M > 0 (isolates the low-frequency
    boundary from the M = 0 one).
    """
    grid = np.logspace(np.log10(kT_lo), np.log10(kT_hi), n_coarse)
    rows = coarse_scan(kA, Km, grid)
    if len(rows) < 4:
        return None

    bracket = None
    for r0, r1 in zip(rows[:-1], rows[1:]):
        if r0[key] > 0.0 > r1[key] or r0[key] < 0.0 < r1[key]:
            if require_Mpos and not (r0["M"] > 0 and r1["M"] > 0):
                continue
            bracket = (r0, r1)
            break
    if bracket is None:
        return None

    r0, r1 = bracket
    lo, hi = r0["kT"], r1["kT"]
    s_lo = np.sign(r0[key])
    guess = (r0["q"], r0["z"])
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        out = mn_at(mid, kA, Km, guess)
        if out is None:
            break
        guess = out["guess"]
        if np.sign(out[key]) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-4:
            break
    return dict(kT_star=np.sqrt(lo * hi), kA=kA, Km=Km, kind=key)


def auto_select(candidates, key="M", require_Mpos=False):
    """First (kA, Km) candidate with a clean `key` transition."""
    for kA, Km in candidates:
        tr = find_transition(kA, Km, key=key, require_Mpos=require_Mpos)
        if tr is not None:
            print(f"  [{key}] using kA={kA:g}, Km={Km:g} -> "
                  f"transition at kT*={tr['kT_star']:.4f}")
            return tr
    print(f"  [{key}] no clean transition found among candidates.")
    return None


# ============================================================
# 2. Ray optimum (single declining mode) -- existing machinery
# ============================================================

def ray_optimum(pt, n):
    """Best doubling time along f = f* + alpha*eta_n, with alpha admissible."""
    kR, kT, kA, kC, Km = pt["kR"], pt["kT"], pt["kA"], pt["kC"], pt["Km"]
    qstar, zstar, fstar, taustar = pt["q"], pt["z"], pt["f"], pt["tau"]
    pars = (kR, kT, kA, kC, Km)
    coef = first_order_coeffs(qstar, zstar, kR, kT, kA, kC, Km)

    theta = np.linspace(0.0, 1.0, 4001)
    eta_max = np.max(np.abs(eta_time(theta, taustar, n, coef)))
    margin = min(fstar, 1.0 - fstar)
    a_star = 0.95 * margin / eta_max

    sol = solve_perturbed_doubling(a_star, n, coef, fstar, pars,
                                   (qstar, zstar, taustar))
    if sol is None:
        return None
    return dict(P=sol["P"], alpha=a_star, q0=sol["q0"], z0=sol["z0"],
                coef=coef, n=n)


# ============================================================
# 3. Global optimum -- direct piecewise-constant optimal control
# ============================================================

def integrate_pwc(f_arr, q0, z0, P, pars, m=3, dense=False):
    """RK4 integrate (q, z, logR) over one cycle theta in [0, 1] (t = P theta).

    f_arr holds K piecewise-constant control values (zero-order hold), each
    interval split into m RK4 substeps.  Fixed-step keeps the NLP smooth.
    """
    kR, kT, kA, kC, Km = pars
    K = len(f_arr)
    nstep = K * m
    dth = 1.0 / nstep

    def rhs(Y, f):
        q, z = Y[0], Y[1]
        return P * np.array([
            F_fun(q, z, f, kR, kT, kA, kC, Km),
            G_fun(q, z, f, kR, kT, kA, kC, Km),
            lam_fun(q, z, f, kR, kT, kA, kC, Km),
        ])

    Y = np.array([q0, z0, 0.0])
    if dense:
        traj = np.empty((nstep + 1, 3))
        traj[0] = Y

    for s in range(nstep):
        f = f_arr[s // m]
        k1 = rhs(Y, f)
        k2 = rhs(Y + 0.5 * dth * k1, f)
        k3 = rhs(Y + 0.5 * dth * k2, f)
        k4 = rhs(Y + dth * k3, f)
        Y = Y + (dth / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if dense:
            traj[s + 1] = Y

    if dense:
        return np.linspace(0.0, 1.0, nstep + 1), traj
    return Y[0], Y[1], Y[2]


def global_optimum(pt, n, K=80, m=3, warm=None):
    """Minimise doubling time over all admissible f(t) in [0, 1].

    Unknowns: K control values + (q0, z0, P).  Constraints q(P)=q0, z(P)=z0,
    R(P)=2.  Warm-started from `warm` (dict f/q0/z0/P) else the ray optimum/BG.
    """
    kR, kT, kA, kC, Km = pt["kR"], pt["kT"], pt["kA"], pt["kC"], pt["Km"]
    qstar, zstar, fstar, taustar = pt["q"], pt["z"], pt["f"], pt["tau"]
    pars = (kR, kT, kA, kC, Km)
    th_mid = (np.arange(K) + 0.5) / K

    if warm is not None:
        f0 = np.clip(np.interp(th_mid, warm["theta"], warm["f"]), 0.0, 1.0)
        q0, z0, P0 = warm["q0"], warm["z0"], warm["P"]
    else:
        ray = ray_optimum(pt, n)
        if ray is not None:
            f0 = np.clip(fstar + ray["alpha"]
                         * eta_time(th_mid, ray["P"], n, ray["coef"]),
                         0.0, 1.0)
            q0, z0, P0 = ray["q0"], ray["z0"], ray["P"]
        else:
            f0 = np.full(K, fstar)
            q0, z0, P0 = qstar, zstar, taustar

    x0 = np.concatenate([f0, [np.log(q0), np.log(z0), np.log(P0)]])
    bounds = ([(0.0, 1.0)] * K
              + [(np.log(1e-3), np.log(1e3)),
                 (np.log(1e-3), np.log(1e3)),
                 (np.log(0.05 * taustar), np.log(20.0 * taustar))])

    def unpack(x):
        return (np.clip(x[:K], 0.0, 1.0),
                np.exp(x[K]), np.exp(x[K + 1]), np.exp(x[K + 2]))

    def obj(x):
        return np.exp(x[K + 2])                 # P (doubling time)

    def obj_jac(x):
        g = np.zeros(K + 3)
        g[K + 2] = np.exp(x[K + 2])
        return g

    def con(x):
        f, q0_, z0_, P_ = unpack(x)
        qK, zK, LK = integrate_pwc(f, q0_, z0_, P_, pars, m=m)
        return np.array([qK - q0_, zK - z0_, LK - LN2])

    res = minimize(obj, x0, jac=obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": con}],
                   options={"maxiter": 250, "ftol": 1e-11})

    f, q0_, z0_, P_ = unpack(res.x)
    resid = float(np.max(np.abs(con(res.x))))
    return dict(P=P_, f=f, q0=q0_, z0=z0_, theta=th_mid,
                feasible=resid < 1e-5, resid=resid)


def count_switches(f, thr=0.5):
    """Cyclic crossings of (f - thr): the bang-bang switch count."""
    s = (f > thr).astype(int)
    return int(np.sum(s != np.roll(s, 1)))


def bang_fraction(f, lo=0.05, hi=0.95):
    return float(np.mean((f < lo) | (f > hi)))


# ============================================================
# 4. Analyse one parameter point
# ============================================================

def analyse_point(pt, K=80, warm=None):
    M, N, taustar = pt["M"], pt["N"], pt["tau"]
    th = (np.arange(K) + 0.5) / K
    fstar = pt["f"]
    flat = np.full(K, fstar)

    rec = dict(kT=pt["kT"], M=M, N=N, criterion=pt["criterion"],
               tau=taustar, f=fstar, lam=pt["lambda"], theta=th)

    n = critical_mode(M, N, taustar)
    if n is None:
        # BG is optimal: every strategy is flat f*
        rec.update(n=None, d2tau=np.nan, P_ray=taustar, P_true=taustar,
                   imp_ray=0.0, imp_true=0.0, switches=0, bang=0.0,
                   k_fund=0, dir_f=flat, ray_f=flat, f_opt=flat, fund_f=flat,
                   feasible=True, warm=None)
        return rec

    d2tau = second_variation(M, N, taustar, n)
    rec.update(n=n, d2tau=d2tau)

    # ---- declining direction (linear) + ray minimum ----
    ray = ray_optimum(pt, n)
    P_ray = np.nan
    ray_warm = None
    dir_f = flat.copy()
    ray_f = flat.copy()
    if ray is not None:
        P_ray = ray["P"]
        eta = eta_time(th, ray["P"], n, ray["coef"])
        ray_f = np.clip(fstar + ray["alpha"] * eta, 0.0, 1.0)
        dir_f = np.clip(fstar + 0.35 * ray["alpha"] * eta, 0.0, 1.0)
        ray_warm = dict(theta=th, f=ray_f, q0=ray["q0"], z0=ray["z0"],
                        P=ray["P"])

    # ---- global optimum (warm-start from sweep neighbour, else ray) ----
    topt = global_optimum(pt, n, K=K, warm=warm if warm is not None else ray_warm)
    feasible = topt["feasible"]
    P_true = topt["P"] if feasible else np.nan
    f_opt = topt["f"] if feasible else ray_f

    fund_f, k_fund, _ = dominant_harmonic(f_opt, th)

    rec.update(
        P_ray=P_ray, P_true=P_true,
        imp_ray=100.0 * (taustar - P_ray) / taustar if np.isfinite(P_ray) else np.nan,
        imp_true=100.0 * (taustar - P_true) / taustar if np.isfinite(P_true) else np.nan,
        switches=count_switches(f_opt) if feasible else np.nan,
        bang=bang_fraction(f_opt) if feasible else np.nan,
        k_fund=k_fund, dir_f=dir_f, ray_f=ray_f, f_opt=f_opt, fund_f=fund_f,
        feasible=feasible,
        warm=dict(theta=th, f=f_opt, q0=topt["q0"], z0=topt["z0"],
                  P=topt["P"]) if feasible else None,
    )
    return rec


# ============================================================
# 5. Sweep across a transition
# ============================================================

def build_sweep(tr, K=80, mults=None):
    kA, Km, kTs, kind = tr["kA"], tr["Km"], tr["kT_star"], tr["kind"]
    if mults is None:
        mults = np.exp(np.linspace(np.log(0.5), np.log(4.0), 13))
    records = []
    guess = (1.0, 1.0)
    warm = None
    for kT in kTs * mults:
        pt = mn_at(kT, kA, Km, guess)
        if pt is None:
            continue
        guess = pt["guess"]
        if not (0.05 < pt["f"] < 0.95) or pt["lambda"] <= 0.05:
            continue
        if not (np.isfinite(pt["M"]) and np.isfinite(pt["N"])):
            continue
        # isolate the low-frequency boundary from the M=0 one
        if kind == "criterion" and pt["M"] <= 0.0:
            continue
        n_try = critical_mode(pt["M"], pt["N"], pt["tau"])
        if n_try is not None and n_try > K // 2:
            print(f"  kT={kT:7.3f}: n_crit={n_try} > K/2, skip (grid too coarse)")
            continue
        print(f"  kT={kT:7.3f}  M={pt['M']:+.3e}  N={pt['N']:+.3e}  "
              f"crit={pt['criterion']:+.3e}  f*={pt['f']:.3f}", end="")
        rec = analyse_point(pt, K=K, warm=warm)
        warm = rec["warm"]
        n_str = "-" if rec["n"] is None else str(rec["n"])
        print(f"  n_crit={n_str}  P_true={rec['P_true']:.4f}  "
              f"sw={rec['switches']}")
        records.append(rec)
    return records


# ============================================================
# 6. Plots
# ============================================================

def fig_strategies(records, tr, outdir):
    """One panel per point: declining dir vs ray min vs global opt vs fundamental."""
    recs = sorted(records, key=lambda r: r[tr["kind"]], reverse=True)  # stable -> deep
    n = len(recs)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows),
                             constrained_layout=True, squeeze=False)
    axflat = axes.ravel()

    for ax, r in zip(axflat, recs):
        th, f0 = r["theta"], r["f"]
        ax.plot(th, r["dir_f"], color="C0", ls="--", lw=1.3,
                label="declining direction (linear)")
        ax.plot(th, r["ray_f"], color="C1", lw=1.6,
                label="ray minimum (single mode)")
        ax.plot(th, r["f_opt"], color="C3", lw=2.2,
                label="global optimum")
        ax.plot(th, r["fund_f"], color="C4", ls=":", lw=1.6,
                label="optimum fundamental mode")
        ax.axhline(f0, color="k", ls=":", lw=0.8)
        ax.axhline(0.0, color="gray", ls=":", lw=0.6)
        ax.axhline(1.0, color="gray", ls=":", lw=0.6)
        ax.set_ylim(-0.08, 1.08)
        if r["n"] is None:
            ttl = "BG optimal (no declining dir)"
        else:
            ttl = (rf"$n_{{\rm crit}}$={r['n']}, sw={r['switches']}, "
                   rf"$2k$={2*r['k_fund']}" + "\n"
                   rf"ray $-${r['imp_ray']:.2f}%, opt $-${r['imp_true']:.2f}%")
        sub = (rf"$M$={r['M']:+.1e}" if tr["kind"] == "M"
               else rf"crit={r['criterion']:+.1e}")
        ax.set_title(sub + "\n" + ttl, fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in axflat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"cycle phase $\theta=t/P$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$f(\theta)$", fontsize=9)
    handles, labels = axflat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Strategy across {KIND_LABEL[tr['kind']]}   "
                 rf"($k_A$={tr['kA']:g}, $K_m$={tr['Km']:g})", fontsize=12)

    png = os.path.join(outdir, f"phase_transition_strategies_{KIND_TAG[tr['kind']]}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {png}")
    return fig


def fig_scaling(records, tr, outdir):
    kind = tr["kind"]
    recs = sorted(records, key=lambda r: r[kind])
    kT = np.array([r["kT"] for r in recs])
    M = np.array([r["M"] for r in recs])
    N = np.array([r["N"] for r in recs])
    crit = np.array([r["criterion"] for r in recs])
    order = np.array([r[kind] for r in recs])
    dist = -order                                   # >0 inside the unstable region
    unstable = np.array([r["n"] is not None for r in recs])
    nmin = np.array([np.nan if r["n"] is None else r["n"] for r in recs], float)
    sw = np.array([r["switches"] for r in recs], float)
    imp_r = np.array([r["imp_ray"] for r in recs])
    imp_t = np.array([r["imp_true"] for r in recs])

    u = unstable & np.isfinite(dist)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(kT, M, "o-", color="C0", label="M")
    ax.plot(kT, crit, "^-", color="C6", label="criterion")
    ax.plot(kT, N, "s-", color="C2", alpha=0.5, label="N")
    ax.axhline(0.0, color="gray", lw=1)
    ax.axvline(tr["kT_star"], color="C3", ls="--",
               label=rf"$k_T^*$={tr['kT_star']:.2f}")
    ax.set_xlabel(r"$k_T$")
    ax.set_ylabel("second-variation coefficients")
    ax.set_title(f"A. Crossing {KIND_LABEL[kind]}", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(dist[u], nmin[u], "o-", color="C4")
    ax.set_xlabel(rf"$-${kind}  (distance into unstable region)")
    ax.set_ylabel(r"critical mode $n_{\rm crit}$")
    ttl = (r"B. $n_{\rm crit}\sim\sqrt{-N/M}$ diverges" if kind == "M"
           else r"B. $n_{\rm crit}=1$ stays finite")
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(dist[u], imp_r[u], "s--", color="C1", label="ray minimum")
    ax.plot(dist[u], imp_t[u], "o-", color="C3", label="global optimum")
    ax.axhline(0.0, color="gray", lw=1)
    ax.set_xlabel(rf"$-${kind}  (distance into unstable region)")
    ax.set_ylabel(r"improvement $100(\tau^*-P)/\tau^*$  [%]")
    ax.set_title("C. Order parameter (gain) vanishes at the transition")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(dist[u], sw[u], "o-", color="C5")
    ax.set_xlabel(rf"$-${kind}  (distance into unstable region)")
    ax.set_ylabel("switches in global optimum")
    ax.set_title("D. Switching structure vs depth")
    ax.grid(True, alpha=0.3)

    fig.suptitle(rf"Scaling across the transition  |  "
                 rf"$k_A$={tr['kA']:g}, $K_m$={tr['Km']:g}", fontsize=13)
    png = os.path.join(outdir, f"phase_transition_scaling_{KIND_TAG[kind]}.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    return fig


def fig_frequency(records, tr, outdir):
    """The frequency link: does n_crit predict the optimum's switch count?"""
    kind = tr["kind"]
    recs = [r for r in records
            if r["n"] is not None and r["feasible"] and np.isfinite(r["switches"])]
    if len(recs) == 0:
        print("  (no unstable feasible points for the frequency figure)")
        return None
    recs = sorted(recs, key=lambda r: r[kind])
    ncrit = np.array([r["n"] for r in recs], float)
    sw = np.array([r["switches"] for r in recs], float)
    twok = np.array([2 * r["k_fund"] for r in recs], float)
    dist = np.array([-r[kind] for r in recs], float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    ax = axes[0]
    lim = max(2.0, np.nanmax([ncrit.max(), sw.max(), twok.max()]) + 1)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label=r"switches $= n_{\rm crit}$")
    ax.plot(ncrit, sw, "o", color="C3", ms=8,
            label="global-optimum switches")
    ax.plot(ncrit, twok, "s", color="C4", ms=7, mfc="none",
            label=r"$2\times$ fundamental index")
    ax.set_xlabel(r"declining mode $n_{\rm crit}$ (from second variation)")
    ax.set_ylabel("frequency of the optimum")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Does the declining mode predict the switching frequency?")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dist, ncrit, "o-", color="C0", label=r"$n_{\rm crit}$ (theory)")
    ax.plot(dist, sw, "s-", color="C3", label="switches (optimum)")
    ax.plot(dist, twok, "^:", color="C4", label=r"$2k_{\rm fund}$ (optimum)")
    ax.set_xlabel(rf"$-${kind}  (distance into unstable region)")
    ax.set_ylabel("mode / switch number")
    ax.set_title("All three frequency measures vs depth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Frequency connection  |  {KIND_LABEL[kind]}", fontsize=12)
    png = os.path.join(outdir, f"phase_transition_frequency_{KIND_TAG[kind]}.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    return fig


# ============================================================
# 7. Run one transition end-to-end
# ============================================================

def run_case(tr, outdir, K=80):
    kind = tr["kind"]
    print(f"\n=== {KIND_LABEL[kind]} ===")
    print(f"Sweeping kT around kT*={tr['kT_star']:.4f} "
          f"(kA={tr['kA']:g}, Km={tr['Km']:g}) ...")
    records = build_sweep(tr, K=K)
    if len(records) < 3:
        print("  too few valid points; skipping figures for this case.")
        return records

    print("\n" + "=" * 84)
    print(f"{'kT':>8} {'M':>11} {'criterion':>11} {'n_crit':>6} "
          f"{'imp_ray%':>9} {'imp_true%':>10} {'switches':>8} {'2k':>4}")
    print("-" * 84)
    for r in sorted(records, key=lambda x: x[kind]):
        ns = "-" if r["n"] is None else f"{r['n']:d}"
        print(f"{r['kT']:8.3f} {r['M']:+11.3e} {r['criterion']:+11.3e} {ns:>6} "
              f"{r['imp_ray']:9.3f} {r['imp_true']:10.3f} "
              f"{r['switches']:>8} {2*r['k_fund']:>4}")
    print("=" * 84)

    fig_strategies(records, tr, outdir)
    fig_scaling(records, tr, outdir)
    fig_frequency(records, tr, outdir)
    return records


# ============================================================
# 8. Main
# ============================================================

def main():
    outdir = os.path.dirname(os.path.abspath(__file__))

    # (A) high-frequency boundary: M = 0, N > 0
    cand_M = [(0.4, 0.1), (0.4, 0.05), (0.6, 0.1),
              (0.25, 0.05), (0.4, 0.01), (0.2, 0.001)]
    # (B) low-frequency boundary: criterion = 0 with M > 0
    cand_C = [(kA, Km)
              for Km in (0.001, 0.01, 0.1, 1.0)
              for kA in (0.1, 0.2, 0.4, 0.8, 1.5, 3.0)]

    print("Locating the high-frequency boundary (M = 0) ...")
    trM = auto_select(cand_M, key="M")
    print("Locating the low-frequency boundary (criterion = 0, M > 0) ...")
    trC = auto_select(cand_C, key="criterion", require_Mpos=True)

    if trM is not None:
        run_case(trM, outdir)
    if trC is not None:
        run_case(trC, outdir)
    if trM is None and trC is None:
        print("No transitions found; widen the candidate lists.")
        return
    plt.show()


if __name__ == "__main__":
    main()
