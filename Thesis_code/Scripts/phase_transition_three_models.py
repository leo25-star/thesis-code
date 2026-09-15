"""
phase_transition_three_models.py
================================

Phase-transition figures for the THREE smooth-limitation models of Section 3
that use the general second-variation derivation of Section 3.2.

Working in the reduced (dimensionless) ratio system of Section 3.2, with
q = T/R, z = A/R, u = k_T/k_R, v = k_C/k_A, control f = f_R:

    q' = phi(q,z) [ (1 - f) - f q ]
    z' = q - phi(q,z) ( u/v + f z )
    (R'/R) = f phi(q,z)                    (=> doubling rate lambda = f phi)

The three models differ only in the saturation factor phi (Sec. 3.4):

  * MM      Michaelis-Menten          phi = z / ( z + kappa (1 + chi u q) )
                transition at  chi u = 1                       
  * modMM   modified MM coefficient    phi = z / ( z + delta + kappa (1 + chi u q) )
                transition at  K_m = 1 / (chi u - 1)           
  * conc    concentration limited      phi = z / ( 1 + chi u q )
                transition at  chi u = 1                        

BG optimality is governed by the reduced second variation
delta^2 tau[y] = int ( M ydot^2 + N y^2 ) dt.  N > 0 throughout, so BG loses
optimality exactly where M changes sign - the phase transition.

For each model the script
  1. fixes  k_A/k_C = 1/v  and  chi  (and, for modMM, k_T/k_R = u), then locates
     the transition numerically by sweeping the model's control parameter:
        MM, conc : sweep u = k_T/k_R   (transition u* = 1/chi)
        modMM    : sweep K_m           (transition K_m* = 1/(chi u - 1))
  2. picks 3 values before the transition (M>0, BG optimal) and 3 after (M<0);
  3. runs a full piecewise-constant optimal-control search at each value;
  4. draws a 2x3 grid of the optimal strategy f(theta) - top row before the
     transition, bottom row after - one figure per model.

Run  `python phase_transition_three_models.py probe`  to print only the M-sweep
tables and located transitions (fast; no optimal-control search).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.integrate import solve_ivp

LN2 = np.log(2.0)

# Parameter tuple layout:  P = (u, v, chi, Km, delta)
IU, IV, ICHI, IKM, IDEL = range(5)


# ============================================================
# 0. Models: phi(q, z) and its analytic first derivatives
# ============================================================

def phi_derivs(model, q, z, P):
    """Return (S, Sq, Sz) = (phi, dphi/dq, dphi/dz) for the chosen model.

    All three share phi = z / D and differ only in the denominator D.
    """
    u, chi, Km, delta = P[IU], P[ICHI], P[IKM], P[IDEL]
    xu = chi * u

    if model == "MM":
        D = z + Km * (1.0 + xu * q)
        S = z / D
        Sq = -z * Km * xu / D**2
        Sz = (Km * (1.0 + xu * q)) / D**2        # = (D - z) / D^2
    elif model == "modMM":
        D = z + delta + Km * (1.0 + xu * q)
        S = z / D
        Sq = -z * Km * xu / D**2
        Sz = (delta + Km * (1.0 + xu * q)) / D**2
    elif model == "conc":
        D = 1.0 + xu * q                          # independent of z
        S = z / D
        Sq = -z * xu / D**2
        Sz = 1.0 / D
    else:
        raise ValueError(f"unknown model {model!r}")
    return S, Sq, Sz


def F_fun(model, q, z, f, P):
    S, _, _ = phi_derivs(model, q, z, P)
    return S * ((1.0 - f) - f * q)


def G_fun(model, q, z, f, P):
    S, _, _ = phi_derivs(model, q, z, P)
    return q - S * (P[IU] / P[IV] + f * z)


def lam_fun(model, q, z, f, P):
    S, _, _ = phi_derivs(model, q, z, P)
    return f * S


# ============================================================
# 1. First-order (linearization) coefficients at a BG point
#    - exactly the reduced coefficients of Section 3.2.
# ============================================================

def first_order_coeffs(model, q, z, P):
    S, Sq, Sz = phi_derivs(model, q, z, P)
    u, v = P[IU], P[IV]

    B = 1.0 + q
    f = 1.0 / B
    C = f                                    # so that S*C = phi* f*
    V0 = u / v + f * z
    m = z / B

    a = 1.0 - V0 * Sq + z * S * f / B
    b = V0 * Sz + S * f
    gamma = a - b * m

    alpha = 1.0 / B - Sq / S
    beta = Sz / S
    stationarity = alpha * b - beta * a

    return dict(S=S, Sq=Sq, Sz=Sz, B=B, f=f, C=C, m=m,
                a=a, b=b, gamma=gamma, alpha=alpha, beta=beta,
                stationarity=stationarity)


def solve_bg(model, P, guess=(1.0, 1.0)):
    """Solve for the BG fixed point: G(q,z,f*)=0 and stationarity=0."""
    def equations(logv):
        q, z = np.exp(logv)
        c = first_order_coeffs(model, q, z, P)
        return np.array([G_fun(model, q, z, c["f"], P), c["stationarity"]])

    sol = root(equations, np.log(guess), method="hybr")
    if not sol.success:
        return None
    q, z = np.exp(sol.x)
    c = first_order_coeffs(model, q, z, P)
    f = c["f"]
    if not (np.isfinite(q) and np.isfinite(z) and np.isfinite(f)):
        return None
    if q <= 0 or z <= 0 or not (0 < f < 1):
        return None
    return q, z, f, c


# ============================================================
# 2. Numerical M, N  (Hessian of H = lambda + p1 F + p2 G)
# ============================================================

def _grad(func, X, h=1e-5):
    g = np.zeros(3)
    for i in range(3):
        step = h * max(1.0, abs(X[i]))
        xp, xm = X.copy(), X.copy()
        xp[i] += step
        xm[i] -= step
        g[i] = (func(xp) - func(xm)) / (2.0 * step)
    return g


def _hess(func, X, h=1e-4):
    H = np.zeros((3, 3))
    f0 = func(X)
    steps = np.array([h * max(1.0, abs(x)) for x in X])
    for i in range(3):
        ei = np.zeros(3)
        ei[i] = steps[i]
        H[i, i] = (func(X + ei) - 2 * f0 + func(X - ei)) / steps[i]**2
        for j in range(i + 1, 3):
            ej = np.zeros(3)
            ej[j] = steps[j]
            H[i, j] = (func(X + ei + ej) - func(X + ei - ej)
                       - func(X - ei + ej) + func(X - ei - ej)) \
                / (4.0 * steps[i] * steps[j])
            H[j, i] = H[i, j]
    return H


def compute_MN(model, P, guess=(1.0, 1.0)):
    bg = solve_bg(model, P, guess=guess)
    if bg is None:
        return None
    q, z, f, c = bg
    X = np.array([q, z, f], dtype=float)

    lamX = lambda Y: lam_fun(model, Y[0], Y[1], Y[2], P)
    FX = lambda Y: F_fun(model, Y[0], Y[1], Y[2], P)
    GX = lambda Y: G_fun(model, Y[0], Y[1], Y[2], P)

    gl, gF, gG = _grad(lamX, X), _grad(FX, X), _grad(GX, X)

    Aadj = np.array([[gF[0], gG[0]], [gF[1], gG[1]]])
    try:
        p1, p2 = np.linalg.solve(Aadj, -np.array([gl[0], gl[1]]))
    except np.linalg.LinAlgError:
        return None

    Hf = gl[2] + p1 * gF[2] + p2 * gG[2]        # control stationarity check

    HH = _hess(lamX, X) + p1 * _hess(FX, X) + p2 * _hess(GX, X)
    Hqq, Hqz, Hzz = HH[0, 0], HH[0, 1], HH[1, 1]
    Hqf, Hzf = HH[0, 2], HH[1, 2]

    lam = lam_fun(model, q, z, f, P)
    tau = LN2 / lam

    S, B, Cc, m, a, b, gamma = (c["S"], c["B"], c["C"], c["m"],
                                c["a"], c["b"], c["gamma"])
    D = S * B
    d = b + S * Cc
    e = S * Cc * b
    Pc = Hqf + m * Hzf
    Qc = b * Hqf + a * Hzf

    A_ydot2 = Hqq + 2 * m * Hqz + m**2 * Hzz
    A_y2 = b**2 * Hqq + 2 * a * b * Hqz + a**2 * Hzz

    M = -(A_ydot2 - (2.0 / D) * (Pc * d - Qc)) / (2.0 * lam * gamma**2)
    N = -(A_y2 - (2.0 / D) * Qc * e) / (2.0 * lam * gamma**2)
    criterion = N + (np.pi**2 / tau**2) * M

    return dict(q=q, z=z, f=f, lam=lam, tau=tau, M=M, N=N,
                criterion=criterion, Hf=Hf, guess=(q, z))


# ============================================================
# 3. Second-variation mode test (phi-independent)
# ============================================================

def y_shape(theta, n):
    """y = sin^2(pi theta) sin(n pi theta) and its first two derivatives."""
    pi = np.pi
    p = np.sin(pi * theta)**2
    dp = pi * np.sin(2 * pi * theta)
    ddp = 2 * pi**2 * np.cos(2 * pi * theta)
    s = np.sin(n * pi * theta)
    ds = n * pi * np.cos(n * pi * theta)
    dds = -(n * pi)**2 * np.sin(n * pi * theta)
    return p * s, dp * s + p * ds, ddp * s + 2 * dp * ds + p * dds


def second_variation(M, N, tau, n, ngrid=20001):
    theta = np.linspace(0.0, 1.0, ngrid)
    y, dy, _ = y_shape(theta, n)
    return M * np.trapz(dy**2, theta) / tau + N * tau * np.trapz(y**2, theta)


def critical_mode(M, N, tau, nmax=80):
    """Smallest mode n with delta^2 tau[y_n] < 0, else None (BG optimal)."""
    if M >= 0.0 and N >= 0.0:
        return None
    for n in range(1, nmax + 1):
        if second_variation(M, N, tau, n) < 0.0:
            return n
    return None


def eta_time(theta, Pd, n, coef):
    """Control perturbation eta(t) generating the mode-n variation y_n."""
    S, B, C, b, gamma = coef["S"], coef["B"], coef["C"], coef["b"], coef["gamma"]
    y, dy, ddy = y_shape(theta, n)
    ydot = dy / Pd
    yddot = ddy / Pd**2
    return -1.0 / (S * B * gamma) * (yddot + (b + S * C) * ydot + S * C * b * y)


# ============================================================
# 4. Ray optimum (best strategy along the single declining mode)
# ============================================================

def _integrate_cycle(model, q0, z0, Pd, f_of_theta, P):
    def rhs(theta, Y):
        q, z, R = Y
        f = float(f_of_theta(theta))
        return [Pd * F_fun(model, q, z, f, P),
                Pd * G_fun(model, q, z, f, P),
                Pd * lam_fun(model, q, z, f, P) * R]
    return solve_ivp(rhs, (0.0, 1.0), [q0, z0, 1.0],
                     method="LSODA", rtol=1e-10, atol=1e-12)


def ray_optimum(model, pt, P, n):
    qs, zs, fs, taus = pt["q"], pt["z"], pt["f"], pt["tau"]
    coef = first_order_coeffs(model, qs, zs, P)

    theta = np.linspace(0.0, 1.0, 4001)
    eta_max = np.max(np.abs(eta_time(theta, taus, n, coef)))
    margin = min(fs, 1.0 - fs)
    alpha = 0.95 * margin / eta_max

    def f_of_theta(th):
        return np.clip(fs + alpha * eta_time(np.atleast_1d(th), taus, n, coef),
                       0.0, 1.0)

    def residual(uu):
        q0, z0, Pd = np.exp(uu)
        sol = _integrate_cycle(model, q0, z0, Pd, f_of_theta, P)
        if not sol.success:
            return [1e3, 1e3, 1e3]
        qP, zP, RP = sol.y[:, -1]
        return [qP - q0, zP - z0, RP - 2.0]

    out = root(residual, np.log([qs, zs, taus]), method="hybr", tol=1e-12)
    if not out.success:
        return None
    q0, z0, Pd = np.exp(out.x)
    return dict(P=Pd, alpha=alpha, q0=q0, z0=z0, coef=coef, n=n)


# ============================================================
# 5. Global optimum -- direct piecewise-constant optimal control
# ============================================================

def _dyn_jac(model, q, z, f, P):
    """Dynamics g=[F,G,lam] plus analytic Jacobians dg/d(q,z,logR) and dg/df."""
    S, Sq, Sz = phi_derivs(model, q, z, P)
    uv = P[IU] / P[IV]

    PF = (1.0 - f) - f * q
    F = S * PF
    Fq = Sq * PF + S * (-f)
    Fz = Sz * PF
    Ff = -S * (1.0 + q)

    PG = uv + f * z
    G = q - S * PG
    Gq = 1.0 - Sq * PG
    Gz = -Sz * PG - S * f
    Gf = -S * z

    lam = f * S
    Lq = f * Sq
    Lz = f * Sz
    Lf = S

    g = np.array([F, G, lam])
    A = np.array([[Fq, Fz, 0.0], [Gq, Gz, 0.0], [Lq, Lz, 0.0]])
    gf = np.array([Ff, Gf, Lf])
    return g, A, gf


def _rk4_step_jac(model, Y, f, Pd, dth, P):
    I3 = np.eye(3)

    g1, A1, gf1 = _dyn_jac(model, Y[0], Y[1], f, P)
    k1 = Pd * g1
    dk1_Y, dk1_f, dk1_P = Pd * A1, Pd * gf1, g1

    Y2 = Y + 0.5 * dth * k1
    g2, A2, gf2 = _dyn_jac(model, Y2[0], Y2[1], f, P)
    k2 = Pd * g2
    J2 = Pd * A2
    dk2_Y = J2 @ (I3 + 0.5 * dth * dk1_Y)
    dk2_f = J2 @ (0.5 * dth * dk1_f) + Pd * gf2
    dk2_P = J2 @ (0.5 * dth * dk1_P) + g2

    Y3 = Y + 0.5 * dth * k2
    g3, A3, gf3 = _dyn_jac(model, Y3[0], Y3[1], f, P)
    k3 = Pd * g3
    J3 = Pd * A3
    dk3_Y = J3 @ (I3 + 0.5 * dth * dk2_Y)
    dk3_f = J3 @ (0.5 * dth * dk2_f) + Pd * gf3
    dk3_P = J3 @ (0.5 * dth * dk2_P) + g3

    Y4 = Y + dth * k3
    g4, A4, gf4 = _dyn_jac(model, Y4[0], Y4[1], f, P)
    k4 = Pd * g4
    J4 = Pd * A4
    dk4_Y = J4 @ (I3 + dth * dk3_Y)
    dk4_f = J4 @ (dth * dk3_f) + Pd * gf4
    dk4_P = J4 @ (dth * dk3_P) + g4

    Y_next = Y + (dth / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    Phi = I3 + (dth / 6.0) * (dk1_Y + 2 * dk2_Y + 2 * dk3_Y + dk4_Y)
    bb = (dth / 6.0) * (dk1_f + 2 * dk2_f + 2 * dk3_f + dk4_f)
    cc = (dth / 6.0) * (dk1_P + 2 * dk2_P + 2 * dk3_P + dk4_P)
    return Y_next, Phi, bb, cc


def _solve_ocp(model, x0, bounds, K, P, m):
    I3 = np.eye(3)
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    nstep = K * m
    dth = 1.0 / nstep

    def unpack(x):
        return (np.clip(x[:K], 0.0, 1.0),
                np.exp(x[K]), np.exp(x[K + 1]), np.exp(x[K + 2]))

    def con_jac_full(x):
        f, q0_, z0_, Pd = unpack(x)
        Phis = np.empty((nstep, 3, 3))
        bs = np.empty((nstep, 3))
        cs = np.empty((nstep, 3))
        Y = np.array([q0_, z0_, 0.0])
        for s in range(nstep):
            Y, Phis[s], bs[s], cs[s] = _rk4_step_jac(model, Y, f[s // m], Pd,
                                                     dth, P)
        cvec = np.array([Y[0] - q0_, Y[1] - z0_, Y[2] - LN2])

        T = I3.copy()
        df_col = np.zeros((3, K))
        dP = np.zeros(3)
        for s in range(nstep - 1, -1, -1):
            df_col[:, s // m] += T @ bs[s]
            dP += T @ cs[s]
            T = T @ Phis[s]
        Psi = T

        Jx = np.zeros((3, K + 3))
        Jx[:, :K] = df_col
        Jx[:, K] = (Psi[:, 0] - e0) * q0_
        Jx[:, K + 1] = (Psi[:, 1] - e1) * z0_
        Jx[:, K + 2] = dP * Pd
        return cvec, Jx

    cache = {"key": None, "c": None, "J": None}

    def ensure(x):
        key = np.asarray(x, float).tobytes()
        if cache["key"] != key:
            c_, J_ = con_jac_full(x)
            cache.update(key=key, c=c_, J=J_)
        return cache

    res = minimize(
        lambda x: np.exp(x[K + 2]),
        x0,
        jac=lambda x: np.concatenate([np.zeros(K + 2), [np.exp(x[K + 2])]]),
        method="SLSQP", bounds=bounds,
        constraints=[{"type": "eq",
                      "fun": lambda x: ensure(x)["c"],
                      "jac": lambda x: ensure(x)["J"]}],
        options={"maxiter": 300, "ftol": 1e-11},
    )
    f, q0_, z0_, Pd = unpack(res.x)
    resid = float(np.max(np.abs(ensure(res.x)["c"])))
    return dict(P=Pd, f=f, q0=q0_, z0=z0_, resid=resid, feasible=resid < 1e-5)


def bangbang_seed(K, k):
    th = (np.arange(K) + 0.5) / K
    return (np.sin(2.0 * np.pi * k * th) > 0.0).astype(float)


def global_optimum(model, pt, P, n, K=80, m=3, warm=None):
    qs, zs, fs, taus = pt["q"], pt["z"], pt["f"], pt["tau"]
    th_mid = (np.arange(K) + 0.5) / K
    bounds = ([(0.0, 1.0)] * K
              + [(np.log(1e-3), np.log(1e3)),
                 (np.log(1e-3), np.log(1e3)),
                 (np.log(0.05 * taus), np.log(20.0 * taus))])

    def make_x0(f0, q0, z0, P0):
        return np.concatenate([np.clip(f0, 0.0, 1.0),
                               [np.log(q0), np.log(z0), np.log(P0)]])

    seeds = []
    if warm is not None:
        f0 = np.interp(th_mid, warm["theta"], warm["f"])
        seeds.append(make_x0(f0, warm["q0"], warm["z0"], warm["P"]))
    ray = ray_optimum(model, pt, P, n) if n is not None else None
    if ray is not None:
        f0 = fs + ray["alpha"] * eta_time(th_mid, ray["P"], n, ray["coef"])
        seeds.append(make_x0(f0, ray["q0"], ray["z0"], ray["P"]))
    seeds.append(make_x0(np.full(K, fs), qs, zs, taus))
    if n is not None:
        for k in sorted({max(1, n - 1), n, n + 1, 2 * n}):
            if 2 * k <= K:
                seeds.append(make_x0(bangbang_seed(K, k), qs, zs, taus))

    best = None
    for x0 in seeds:
        sol = _solve_ocp(model, x0, bounds, K, P, m)
        better = (best is None
                  or (sol["feasible"] and not best["feasible"])
                  or (sol["feasible"] == best["feasible"] and sol["P"] < best["P"]))
        if better:
            best = sol
    best["theta"] = th_mid
    best["ray"] = ray
    return best


def count_switches(f, thr):
    s = (f > thr).astype(int)
    return int(np.sum(s != np.roll(s, 1)))


# ============================================================
# 6. Locate the transition and build the sweep
# ============================================================

SWEEP_IDX = {"u": IU, "Km": IKM}


def set_param(P, key, value):
    P = list(P)
    P[SWEEP_IDX[key]] = value
    return tuple(P)


def sweep_value(P, key):
    return P[SWEEP_IDX[key]]


def mn_at(model, P, key, value, guess):
    out = compute_MN(model, set_param(P, key, value), guess=guess)
    if out is not None:
        out["sweep"] = value
    return out


def coarse_scan(model, P, key, grid):
    rows, guess = [], (1.0, 1.0)
    for val in grid:
        out = mn_at(model, P, key, val, guess)
        if out is None:
            continue
        guess = out["guess"]
        if not (np.isfinite(out["M"]) and np.isfinite(out["N"])):
            continue
        if not (0.05 < out["f"] < 0.95) or out["lam"] <= 0.05:
            continue
        if abs(out["Hf"]) > 1e-2:
            continue
        rows.append(out)
    return rows


def find_transition(model, P, key, lo, hi, n_coarse=40):
    """Bracket and bisect the swept value where M changes sign (M>0 -> M<0)."""
    grid = np.logspace(np.log10(lo), np.log10(hi), n_coarse)
    rows = coarse_scan(model, P, key, grid)
    if len(rows) < 3:
        return None, rows

    bracket = None
    for r0, r1 in zip(rows[:-1], rows[1:]):
        if r0["M"] * r1["M"] < 0.0:
            bracket = (r0, r1)
            break
    if bracket is None:
        return None, rows

    r0, r1 = bracket
    a, b = r0["sweep"], r1["sweep"]
    s_a = np.sign(r0["M"])
    guess = (r0["q"], r0["z"])
    for _ in range(60):
        mid = np.sqrt(a * b)
        out = mn_at(model, P, key, mid, guess)
        if out is None:
            break
        guess = out["guess"]
        if np.sign(out["M"]) == s_a:
            a = mid
        else:
            b = mid
        if b / a - 1.0 < 1e-6:
            break
    return np.sqrt(a * b), rows


# multiplicative offsets from the transition value: 3 before, 3 after
MULTS = np.array([0.55, 0.72, 0.88, 1.13, 1.4, 1.9])


def build_sweep(model, P, key, star, K=80, verbose=True):
    records, guess, warm = [], (1.0, 1.0), None
    for val in star * MULTS:
        pt = mn_at(model, P, key, val, guess)
        if pt is None:
            continue
        guess = pt["guess"]
        if not (0.05 < pt["f"] < 0.95) or pt["lam"] <= 0.05:
            continue
        Pv = set_param(P, key, val)
        n = critical_mode(pt["M"], pt["N"], pt["tau"])
        K_pt = K if n is None else int(min(360, max(K, 6 * n)))
        if verbose:
            print(f"    {key}={val:9.4f}  M={pt['M']:+.3e}  N={pt['N']:+.3e}  "
                  f"f*={pt['f']:.3f}  n_crit={'-' if n is None else n}  K={K_pt}",
                  end="", flush=True)
        rec = analyse_point(model, Pv, pt, n, key, val, K=K_pt, warm=warm)
        warm = rec["warm"]
        if verbose:
            print(f"  ->  P_true={rec['P_true']:.4f}  gain={rec['gain']:.2f}%  "
                  f"switches={rec['switches']}")
        records.append(rec)
    return records


def analyse_point(model, P, pt, n, key, val, K=80, warm=None):
    fs, taus = pt["f"], pt["tau"]
    th = (np.arange(K) + 0.5) / K
    flat = np.full(K, fs)
    rec = dict(sweep_val=val, M=pt["M"], N=pt["N"], tau=taus, f=fs,
               theta=th, n=n,
               Pvec=P,                               # full parameter tuple
               q_bg=pt["q"], z_bg=pt["z"])           # BG fixed point

    if n is None:                                    # BG optimal: flat f*
        rec.update(dir_f=flat, f_opt=flat, P_true=taus, gain=0.0,
                   switches=0, feasible=True, warm=None,
                   q0=pt["q"], z0=pt["z"], P_cycle=taus)
        return rec

    top = global_optimum(model, pt, P, n, K=K, warm=warm)
    feasible = top["feasible"]
    P_true = top["P"] if feasible else np.nan
    f_opt = top["f"] if feasible else flat
    gain = 100.0 * (taus - P_true) / taus if np.isfinite(P_true) else 0.0

    dir_f = flat.copy()
    ray = top.get("ray")
    if ray is not None:
        eta = eta_time(th, ray["P"], n, ray["coef"])
        dir_f = np.clip(fs + 0.35 * ray["alpha"] * eta, 0.0, 1.0)

    rec.update(dir_f=dir_f, f_opt=f_opt, P_true=P_true, gain=gain,
               switches=count_switches(f_opt, fs) if feasible else np.nan,
               feasible=feasible,
               # initial condition and period of the cycle actually integrated
               # (falls back to the BG point when the search did not converge)
               q0=top["q0"] if feasible else pt["q"],
               z0=top["z0"] if feasible else pt["z"],
               P_cycle=top["P"] if feasible else taus,
               warm=dict(theta=th, f=f_opt, q0=top["q0"], z0=top["z0"],
                         P=top["P"]) if feasible else None)
    return rec


# ============================================================
# 7. Plot: 2x3 grid of optimal strategies (before / after)
# ============================================================

def fig_strategies(model, cfg, P, star, records, outdir):
    key = cfg["sweep"]
    recs = sorted(records, key=lambda r: r["M"], reverse=True)
    before = sorted([r for r in recs if r["M"] > 0.0][-3:],
                    key=lambda r: r["sweep_val"])
    after = sorted([r for r in recs if r["M"] <= 0.0][:3],
                   key=lambda r: r["sweep_val"])

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
            th = r["theta"]
            ax.plot(th, r["dir_f"], color="C0", ls="--", lw=1.3,
                    label="declining direction")
            ax.plot(th, r["f_opt"], color="C3", lw=2.2, label="global optimum")
            ax.axhline(r["f"], color="k", ls=":", lw=0.8)
            ax.axhline(0.0, color="gray", ls=":", lw=0.5)
            ax.axhline(1.0, color="gray", ls=":", lw=0.5)
            ax.set_ylim(-0.08, 1.08)
            ax.set_title(rf"${sweep_sym}$={r['sweep_val']:.3g},  "
                         rf"gain={r['gain']:.2f}%", fontsize=12)
            ax.tick_params(labelsize=8)

    for i in range(2):
        axes[i, 0].annotate(row_label[i], xy=(0, 0.5), xytext=(-0.52, 0.5),
                            xycoords="axes fraction", textcoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=10, fontweight="bold")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"cycle phase $\theta=t/\tau$", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"allocation $f_R(\theta)$", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    kA_kC = 1.0 / P[IV]
    if key == "u":
        trans = rf"transition at $u=1/\chi={star:.3g}$"
        fixed = rf"$v={1/kA_kC:g}$, $\chi={P[ICHI]:g}$"
        if model != "conc":       # phi of the concentration-limited model has no K_m
            fixed += rf", $K_m={P[IKM]:g}$"
    else:
        trans = rf"transition at $K_m={star:.3g}$"
        fixed = rf"$v={1/kA_kC:g}$, $\chi={P[ICHI]:g}$, $u={P[IU]:g}$"
    fig.suptitle(f"{cfg['title']}:  optimal strategy across the phase transition\n"
                 f"({trans};  {fixed})", fontsize=15)

    png = os.path.join(outdir, f"phase_transition_{cfg['tag']}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {png}")
    return fig


# ============================================================
# 8. Model configurations and driver
# ============================================================

# Fixed choices:  k_A/k_C = 0.4  (=> v = 2.5),  chi = 1.5.
CHI = 1.5
V = 2.5

MODELS = {
    "MM": dict(
        title="Michaelis-Menten model",
        tag="MM_model",
        sweep="u",                        # transition at u* = 1/chi
        P=(None, V, CHI, 1.0, 0.0),       # u swept; Km=kappa=1
        lo=0.1, hi=6.0,
    ),
    "modMM": dict(
        title="Modified Michaelis-Menten model",
        tag="modMM_model",
        sweep="Km",                       # transition K_m = 1/(chi u - 1)
        P=(1.0, V, CHI, None, 1.0),       # u=1 (chi u = 1.5); Km swept; delta=1
        lo=0.05, hi=60.0,
    ),
    "conc": dict(
        title="Concentration-limited model",
        tag="conc_model",
        sweep="u",                        # transition at u* = 1/chi
        P=(None, V, CHI, 1.0, 0.0),
        lo=0.1, hi=6.0,
    ),
}


def run_model(name, cfg, outdir, probe=False):
    key = cfg["sweep"]
    start = np.sqrt(cfg["lo"] * cfg["hi"])
    P = set_param(cfg["P"], key, start)

    print(f"\n=== {cfg['title']}  (sweep {key}) ===")
    star, rows = find_transition(name, P, key, cfg["lo"], cfg["hi"])
    if star is None:
        print("  No M sign-change found in the scanned range.")
        for r in rows:
            print(f"    {key}={r['sweep']:9.4f}  M={r['M']:+.3e}  "
                  f"N={r['N']:+.3e}  f*={r['f']:.3f}")
        return None

    pred = (1.0 / P[ICHI] if key == "u"
            else 1.0 / (P[ICHI] * P[IU] - 1.0))
    print(f"  transition (numeric): {key}* = {star:.4f}   "
          f"(analytic prediction {pred:.4f})")
    if probe:
        return star

    records = build_sweep(name, P, key, star)
    if len(records) < 2:
        print("  Too few valid sweep points for a figure.")
        return star
    fig_strategies(name, cfg, P, star, records, outdir)

    # companion figure: the same optimal solutions drawn in the (q,z) plane
    # (imported lazily so this script still runs if the module is absent)
    try:
        from qz_space_three_models import fig_qz_space
    except ImportError as exc:
        print(f"  [qz_space_three_models unavailable: {exc}]")
    else:
        fig_qz_space(name, cfg, P, star, records, outdir)
    return star


def main():
    # Figures are written to Thesis_code/PNGs (one level up from Scripts/)
    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PNGs"
    )
    os.makedirs(outdir, exist_ok=True)
    probe = len(sys.argv) > 1 and sys.argv[1] == "probe"
    for name, cfg in MODELS.items():
        run_model(name, cfg, outdir, probe=probe)
    if not probe:
        plt.show()


if __name__ == "__main__":
    main()
