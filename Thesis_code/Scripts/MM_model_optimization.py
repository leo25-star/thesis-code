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

This script locates both boundaries (by sweeping K_m at fixed k_T, k_A), and at
each point builds and graphs:

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
from Calculate_second_variation import (  # noqa: E402
    F_fun, G_fun, lam_fun, compute_MN, first_order_coeffs,
)
from Find_declining_direction import (  # noqa: E402
    eta_time, second_variation, solve_perturbed_doubling,
)

LN2 = np.log(2.0)

KIND_TAG = {"M": "Mzero", "criterion": "criterion"}
KIND_LABEL = {
    "M": "M = 0",
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
# 1. Locate a transition in K_m (generic over the order parameter)
# ============================================================

def mn_at(kT, kA, Km, guess, kR=1.0, kC=1.0):
    """compute_MN wrapper returning a tidy record (or None)."""
    out = compute_MN(kR, kT, kA, kC, Km, guess=guess)
    if out is None:
        return None
    out["kT"], out["kA"], out["Km"], out["kR"], out["kC"] = kT, kA, Km, kR, kC
    return out


def coarse_scan(kA, kT, Km_grid):
    """Scan K_m at fixed (kA, kT) and keep well-behaved BG points."""
    rows = []
    guess = (1.0, 1.0)
    for Km in Km_grid:
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


def find_transition(kA, kT, key="M", Km_lo=1e-3, Km_hi=1e2, n_coarse=30,
                    require_Mpos=False):
    """Bracket and bisect the K_m where `key` changes sign (stable -> unstable).

    Convention: stable side has key > 0, unstable side key < 0.
    require_Mpos keeps the bracket inside M > 0 (isolates the low-frequency
    boundary from the M = 0 one).
    """
    grid = np.logspace(np.log10(Km_lo), np.log10(Km_hi), n_coarse)
    rows = coarse_scan(kA, kT, grid)
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
    lo, hi = r0["Km"], r1["Km"]
    s_lo = np.sign(r0[key])
    guess = (r0["q"], r0["z"])
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        out = mn_at(kT, kA, mid, guess)
        if out is None:
            break
        guess = out["guess"]
        if np.sign(out[key]) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-4:
            break
    return dict(Km_star=np.sqrt(lo * hi), kA=kA, kT=kT, kind=key)


def auto_select(candidates, key="M", require_Mpos=False):
    """First (kA, kT) candidate with a clean `key` transition in K_m."""
    for kA, kT in candidates:
        tr = find_transition(kA, kT, key=key, require_Mpos=require_Mpos)
        if tr is not None:
            print(f"  [{key}] using kA={kA:g}, kT={kT:g} -> "
                  f"transition at Km*={tr['Km_star']:.4f}")
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


def bangbang_seed(K, k):
    """Square wave in {0, 1} with 2k switches over the cycle (k full cycles).

    A k-harmonic bang-bang control is what the optimum looks like when the
    declining mode has index n_crit = k: it switches 2k times per period.  Used
    as an initial guess so the local NLP solver can *find* the high-frequency
    chattering branch instead of relaxing to the nearby smooth minimum.
    """
    th = (np.arange(K) + 0.5) / K
    return (np.sin(2.0 * np.pi * k * th) > 0.0).astype(float)


def _dyn_jac(q, z, f, pars):
    """Dynamics g=[F,G,lam] and its analytic Jacobian at state (q,z), control f.

    Returns (g, A, gf) where
      g  : (3,)   right-hand side [F, G, lam] (without the P scaling),
      A  : (3,3)  dg/d(q,z,logR) -- the logR column is zero (g is R-free),
      gf : (3,)   dg/df.
    All partials are closed form (F,G,lam are algebraic in S=z/D).
    """
    kR, kT, kA, kC, Km = pars
    D = z + 1.0 + Km * (1.0 + q)
    S = z / D
    Sq = -Km * z / D**2
    Sz = (D - z) / D**2                    # = (1 + Km(1+q)) / D^2

    PF = kT * (1.0 - f) - kR * f * q       # F = S * PF
    F = S * PF
    Fq = Sq * PF + S * (-kR * f)
    Fz = Sz * PF
    Ff = -S * (kT + kR * q)

    PG = kC + f * kR * z                    # G = kA q - S * PG
    G = kA * q - S * PG
    Gq = kA - Sq * PG
    Gz = -Sz * PG - S * (f * kR)
    Gf = -S * (kR * z)

    lam = f * kR * S                        # lam = f kR S
    Lq = f * kR * Sq
    Lz = f * kR * Sz
    Lf = kR * S

    g = np.array([F, G, lam])
    A = np.array([[Fq, Fz, 0.0],
                  [Gq, Gz, 0.0],
                  [Lq, Lz, 0.0]])
    gf = np.array([Ff, Gf, Lf])
    return g, A, gf


def _rk4_step_jac(Y, f, P, dth, pars):
    """One RK4 step of (q,z,logR) plus the Jacobians of the step.

    Reproduces the state update of `integrate_pwc` exactly and, alongside it,
    differentiates the step to yield
      Phi = dY_next/dY   (3,3),
      b   = dY_next/df   (3,),
      c   = dY_next/dP   (3,).
    rhs = P * g, so d rhs/dY = P*A, d rhs/df = P*gf, d rhs/dP = g.
    """
    I3 = np.eye(3)

    g1, A1, gf1 = _dyn_jac(Y[0], Y[1], f, pars)
    k1 = P * g1
    dk1_Y = P * A1
    dk1_f = P * gf1
    dk1_P = g1

    Y2 = Y + 0.5 * dth * k1
    g2, A2, gf2 = _dyn_jac(Y2[0], Y2[1], f, pars)
    k2 = P * g2
    J2 = P * A2
    dk2_Y = J2 @ (I3 + 0.5 * dth * dk1_Y)
    dk2_f = J2 @ (0.5 * dth * dk1_f) + P * gf2
    dk2_P = J2 @ (0.5 * dth * dk1_P) + g2

    Y3 = Y + 0.5 * dth * k2
    g3, A3, gf3 = _dyn_jac(Y3[0], Y3[1], f, pars)
    k3 = P * g3
    J3 = P * A3
    dk3_Y = J3 @ (I3 + 0.5 * dth * dk2_Y)
    dk3_f = J3 @ (0.5 * dth * dk2_f) + P * gf3
    dk3_P = J3 @ (0.5 * dth * dk2_P) + g3

    Y4 = Y + dth * k3
    g4, A4, gf4 = _dyn_jac(Y4[0], Y4[1], f, pars)
    k4 = P * g4
    J4 = P * A4
    dk4_Y = J4 @ (I3 + dth * dk3_Y)
    dk4_f = J4 @ (dth * dk3_f) + P * gf4
    dk4_P = J4 @ (dth * dk3_P) + g4

    Y_next = Y + (dth / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    Phi = I3 + (dth / 6.0) * (dk1_Y + 2 * dk2_Y + 2 * dk3_Y + dk4_Y)
    b = (dth / 6.0) * (dk1_f + 2 * dk2_f + 2 * dk3_f + dk4_f)
    c = (dth / 6.0) * (dk1_P + 2 * dk2_P + 2 * dk3_P + dk4_P)
    return Y_next, Phi, b, c


def _solve_ocp(x0, bounds, K, pars, m):
    """One SLSQP solve of the direct transcription from a single seed x0.

    The equality-constraint Jacobian is supplied analytically: a single forward
    pass builds the per-step transition matrices, then a backward accumulation
    turns them into d(constraint)/d(unknowns).  This is O(K) per Jacobian versus
    SLSQP's O(K) full re-integrations (=> O(K^2)) for a finite-difference one,
    which is what makes the many-interval solves near the transition tractable.
    """
    I3 = np.eye(3)
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    nstep = K * m
    dth = 1.0 / nstep

    def unpack(x):
        return (np.clip(x[:K], 0.0, 1.0),
                np.exp(x[K]), np.exp(x[K + 1]), np.exp(x[K + 2]))

    def _con_and_jac(x):
        f, q0_, z0_, P_ = unpack(x)
        Phis = np.empty((nstep, 3, 3))
        bs = np.empty((nstep, 3))
        cs = np.empty((nstep, 3))
        Y = np.array([q0_, z0_, 0.0])
        for s in range(nstep):
            Y, Phis[s], bs[s], cs[s] = _rk4_step_jac(Y, f[s // m], P_, dth, pars)
        cvec = np.array([Y[0] - q0_, Y[1] - z0_, Y[2] - LN2])

        # Backward accumulation: T = Phi_{N-1} ... Phi_{s+1}
        T = I3.copy()
        df_col = np.zeros((3, K))
        dP = np.zeros(3)
        for s in range(nstep - 1, -1, -1):
            df_col[:, s // m] += T @ bs[s]
            dP += T @ cs[s]
            T = T @ Phis[s]
        Psi = T                                    # dY_final / dY_0

        Jx = np.zeros((3, K + 3))
        Jx[:, :K] = df_col                          # d c / d f_j
        Jx[:, K] = (Psi[:, 0] - e0) * q0_           # via q0 = exp(x[K])
        Jx[:, K + 1] = (Psi[:, 1] - e1) * z0_       # via z0 = exp(x[K+1])
        Jx[:, K + 2] = dP * P_                       # via P  = exp(x[K+2])
        return cvec, Jx

    cache = {"key": None, "c": None, "J": None}

    def _ensure(x):
        x = np.asarray(x, float)
        key = x.tobytes()
        if cache["key"] != key:
            c_, J_ = _con_and_jac(x)
            cache.update(key=key, c=c_, J=J_)
        return cache

    def obj(x):
        return np.exp(x[K + 2])                 # P (doubling time)

    def obj_jac(x):
        g = np.zeros(K + 3)
        g[K + 2] = np.exp(x[K + 2])
        return g

    def con(x):
        return _ensure(x)["c"]

    def con_jac(x):
        return _ensure(x)["J"]

    res = minimize(obj, x0, jac=obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": con, "jac": con_jac}],
                   options={"maxiter": 300, "ftol": 1e-11})
    f, q0_, z0_, P_ = unpack(res.x)
    resid = float(np.max(np.abs(con(res.x))))
    return dict(P=P_, f=f, q0=q0_, z0=z0_, resid=resid, feasible=resid < 1e-5)


def global_optimum(pt, n, K=80, m=3, warm=None):
    """Minimise doubling time over all admissible f(t) in [0, 1].

    Unknowns: K control values + (q0, z0, P).  Constraints q(P)=q0, z(P)=z0,
    R(P)=2.

    Multi-start: SLSQP is a *local* solver, so a smooth warm-start settles into
    the nearest smooth minimum and never discovers the high-frequency
    (chattering) branch that dominates near the M = 0 boundary.  We therefore
    seed it from several guesses -- the sweep neighbour, the smooth ray optimum,
    flat BG, and BANG-BANG square waves at the critical frequency n_crit (and
    twice it) -- then keep the best feasible result.  This is what lets the
    optimum resolve the many-switch / chattering structure.
    """
    kR, kT, kA, kC, Km = pt["kR"], pt["kT"], pt["kA"], pt["kC"], pt["Km"]
    qstar, zstar, fstar, taustar = pt["q"], pt["z"], pt["f"], pt["tau"]
    pars = (kR, kT, kA, kC, Km)
    th_mid = (np.arange(K) + 0.5) / K

    bounds = ([(0.0, 1.0)] * K
              + [(np.log(1e-3), np.log(1e3)),
                 (np.log(1e-3), np.log(1e3)),
                 (np.log(0.05 * taustar), np.log(20.0 * taustar))])

    def make_x0(f0, q0, z0, P0):
        return np.concatenate([np.clip(f0, 0.0, 1.0),
                               [np.log(q0), np.log(z0), np.log(P0)]])

    # ---- assemble seeds ----
    seeds = []
    if warm is not None:
        f0 = np.interp(th_mid, warm["theta"], warm["f"])
        seeds.append(make_x0(f0, warm["q0"], warm["z0"], warm["P"]))
    ray = ray_optimum(pt, n)
    if ray is not None:
        f0 = fstar + ray["alpha"] * eta_time(th_mid, ray["P"], n, ray["coef"])
        seeds.append(make_x0(f0, ray["q0"], ray["z0"], ray["P"]))
    seeds.append(make_x0(np.full(K, fstar), qstar, zstar, taustar))
    # bang-bang seeds at (and around) the critical switching frequency
    for k in sorted({max(1, n - 1), n, n + 1, 2 * n}):
        if 2 * k <= K:                          # grid must resolve 2k switches
            seeds.append(make_x0(bangbang_seed(K, k), qstar, zstar, taustar))

    # ---- solve from each seed; keep the best feasible (else least-infeasible) ----
    best = None
    for x0 in seeds:
        sol = _solve_ocp(x0, bounds, K, pars, m)
        better = (best is None
                  or (sol["feasible"] and not best["feasible"])
                  or (sol["feasible"] == best["feasible"] and sol["P"] < best["P"]))
        if better:
            best = sol

    best["theta"] = th_mid
    return best


def count_switches(f, thr=0.5):
    """Cyclic crossings of (f - thr): the switch count.

    Pass thr = f* (the singular-arc value) rather than 0.5: near the M = 0
    boundary the optimum is a small-amplitude, high-frequency ripple ABOUT f*
    that never reaches 0 or 1, so crossings of 0.5 miss it entirely and the
    chattering looks like zero switches.  Crossings of f* count both that ripple
    and every full bang-bang switch (since 0 < f* < 1).
    """
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

    rec = dict(kT=pt["kT"], Km=pt["Km"], M=M, N=N, criterion=pt["criterion"],
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
        switches=count_switches(f_opt, thr=fstar) if feasible else np.nan,
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

def build_sweep(tr, K=80, Kmax=400, mults=None):
    kA, kT, Kms, kind = tr["kA"], tr["kT"], tr["Km_star"], tr["kind"]
    if mults is None:
        mults = np.exp(np.linspace(np.log(0.25), np.log(4.0), 6))
    records = []
    guess = (1.0, 1.0)
    warm = None
    for Km in Kms * mults:
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
        # Adapt the control resolution to the critical frequency instead of
        # skipping: near the M=0 boundary n_crit ~ sqrt(-N/M) diverges
        # (chattering), so we need several intervals per switch to resolve it.
        n_try = critical_mode(pt["M"], pt["N"], pt["tau"])
        K_pt = K
        if n_try is not None:
            K_pt = int(min(Kmax, max(K, 6 * n_try)))
            if n_try > Kmax // 2:
                print(f"  Km={Km:8.4f}: n_crit={n_try} > Kmax/2; using K={Kmax} "
                      f"(chattering only partially resolved)")
        print(f"  Km={Km:8.4f}  M={pt['M']:+.3e}  N={pt['N']:+.3e}  "
              f"crit={pt['criterion']:+.3e}  f*={pt['f']:.3f}  K={K_pt}", end="")
        rec = analyse_point(pt, K=K_pt, warm=warm)
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
    """2x3 panel: declining direction vs global optimum.

    Top row = three points BEFORE the transition (stable / BG side, order
    parameter > 0); bottom row = three points AFTER it (unstable side).  Each
    panel caption reports only K_m and the gain of the fastest strategy (the
    global optimum) over the balanced-growth (BG) solution.
    """
    kind = tr["kind"]
    # stable -> deep; convention: order parameter > 0 stable, < 0 unstable
    recs = sorted(records, key=lambda r: r[kind], reverse=True)
    before = [r for r in recs if r[kind] > 0.0][-3:]   # three closest before
    after = [r for r in recs if r[kind] <= 0.0][:3]    # three closest after

    ncols = 3
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 6.0),
                             constrained_layout=True, squeeze=False)

    row_recs = [before, after]
    row_label = ["before transition", "after transition"]

    for i in range(2):
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(row_recs[i]):
                ax.axis("off")
                continue
            r = row_recs[i][j]
            th, f0 = r["theta"], r["f"]
            ax.plot(th, r["dir_f"], color="C0", ls="--", lw=1.3,
                    label="declining tau direction")
            ax.plot(th, r["f_opt"], color="C3", lw=2.2,
                    label="global optimum")
            ax.axhline(f0, color="k", ls=":", lw=0.8)
            ax.axhline(0.0, color="gray", ls=":", lw=0.6)
            ax.axhline(1.0, color="gray", ls=":", lw=0.6)
            ax.set_ylim(-0.08, 1.08)
            # gain of the fastest strategy (global optimum) over BG
            gain = r["imp_true"] if np.isfinite(r["imp_true"]) else 0.0
            ax.set_title(rf"$K_m$={r['Km']:.3g},  gain={gain:.2f}%", fontsize=9)
            ax.tick_params(labelsize=8)

    # side captions, one per row (on the leftmost panel of each row)
    for i in range(2):
        axes[i, 0].annotate(
            row_label[i], xy=(0.0, 0.5), xytext=(-0.42, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"cycle phase $\theta=t/\tau$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$f(\theta)$", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Strategy across {KIND_LABEL[kind]}   "
                 rf"($k_A/k_C$={tr['kA']:g}, $k_T/k_R$={tr['kT']:g})", fontsize=12)

    png = os.path.join(outdir, f"phase_transition_strategies_{KIND_TAG[tr['kind']]}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {png}")
    return fig


def fig_MN(records, tr, outdir):
    """M and N vs K_m: the transition occurs at M=0 with N>0 throughout."""
    kind = tr["kind"]
    recs = sorted(records, key=lambda r: r["Km"])
    Km = np.array([r["Km"] for r in recs])
    M = np.array([r["M"] for r in recs])
    N = np.array([r["N"] for r in recs])

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(Km, M, "o-", color="C0", label=r"$M$  (Legendre--Clebsch coeff.)")
    ax.plot(Km, N, "s-", color="C2", label=r"$N$")
    ax.axhline(0.0, color="gray", lw=1)
    ax.axvline(tr["Km_star"], color="C3", ls="--",
               label=rf"$K_m^*$={tr['Km_star']:.3g}  ($M=0$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K_m$")
    ax.set_ylabel("second-variation coefficients")
    ax.set_title(r"Phase transition at $M=0$ with $N>0$ throughout"
                 rf"  |  $k_A$={tr['kA']:g}, $k_T$={tr['kT']:g}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    png = os.path.join(outdir, f"phase_transition_MN_{KIND_TAG[kind]}.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    return fig


def fig_scaling(records, tr, outdir):
    kind = tr["kind"]
    recs = sorted(records, key=lambda r: r[kind])
    Km = np.array([r["Km"] for r in recs])
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
    ax.plot(Km, M, "o-", color="C0", label="M")
    ax.plot(Km, crit, "^-", color="C6", label="criterion")
    ax.plot(Km, N, "s-", color="C2", alpha=0.5, label="N")
    ax.axhline(0.0, color="gray", lw=1)
    ax.axvline(tr["Km_star"], color="C3", ls="--",
               label=rf"$K_m^*$={tr['Km_star']:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K_m$")
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
                 rf"$k_A$={tr['kA']:g}, $k_T$={tr['kT']:g}", fontsize=13)
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
    print(f"Sweeping Km around Km*={tr['Km_star']:.4f} "
          f"(kA={tr['kA']:g}, kT={tr['kT']:g}) ...")
    records = build_sweep(tr, K=K)
    if len(records) < 3:
        print("  too few valid points; skipping figures for this case.")
        return records

    print("\n" + "=" * 84)
    print(f"{'Km':>10} {'M':>11} {'criterion':>11} {'n_crit':>6} "
          f"{'imp_ray%':>9} {'imp_true%':>10} {'switches':>8} {'2k':>4}")
    print("-" * 84)
    for r in sorted(records, key=lambda x: x[kind]):
        ns = "-" if r["n"] is None else f"{r['n']:d}"
        print(f"{r['Km']:10.4f} {r['M']:+11.3e} {r['criterion']:+11.3e} {ns:>6} "
              f"{r['imp_ray']:9.3f} {r['imp_true']:10.3f} "
              f"{r['switches']:>8} {2*r['k_fund']:>4}")
    print("=" * 84)

    fig_strategies(records, tr, outdir)
    fig_MN(records, tr, outdir)
    fig_scaling(records, tr, outdir)
    fig_frequency(records, tr, outdir)
    return records


# ============================================================
# 8. Main
# ============================================================

def main():
    outdir = os.path.dirname(os.path.abspath(__file__))

    # Candidates are (kA, kT); the transition is located by sweeping K_m.
    # (A) high-frequency boundary: M = 0, N > 0  (needs large-ish k_T)
    cand_M = [(kA, kT)
              for kT in (10.0, 20.0, 50.0, 100.0, 200.0)
              for kA in (0.4, 0.2, 0.6, 0.25, 0.1)]
    # (B) low-frequency boundary: criterion = 0 with M > 0
    cand_C = [(kA, kT)
              for kT in (2.0, 5.0, 10.0, 20.0, 50.0)
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
