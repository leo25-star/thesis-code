"""
MM_pmp_optimal.py
=================

Find the *optimal* cyclic growth strategy for the Michaelis-Menten allocation
model via Pontryagin's Minimum Principle (PMP).

PMP structure
-------------
Minimise the doubling time of a cyclic strategy:

    minimise  P = int_0^P 1 dt
    states    q, z  (ratios T/R, A/R)  and  w  with  wdot = lambda
    dynamics  qdot = S(q,z) [ kT(1-f) - kR f q ]
              zdot = kA q - S(q,z) [ kC + f kR z ]
              wdot = lambda = f kR S(q,z)
    control   f(t) in [0, 1]
    cyclic    q(P) = q(0),  z(P) = z(0)        (division resets ratios)
    doubling  w(0) = 0,  w(P) = ln 2           (cell exactly doubles)

The Hamiltonian is

    H = 1 + p_q qdot + p_z zdot + p_w lambda .

Because qdot, zdot, lambda are all AFFINE in f, H is linear in f:

    H = H0(q,z,p) + f * phi(q,z,p),

with the SWITCHING FUNCTION

    phi = d H / d f
        = p_q (-S B) + p_z (-S kR z) + p_w (kR S)
        = S ( -p_q B - p_z kR z + p_w kR ),     B = kT + kR q.

Pontryagin's minimum principle (we minimise H over f in [0,1]) gives

    f* = 0          where phi > 0    (bang-0)
    f* = 1          where phi < 0    (bang-1)
    f* singular     where phi == 0 on an interval   (singular arc).

Costates:  pdot = -dH/dx,  with p_w constant (w is absent from H), and for the
free, autonomous final-time problem H == 0 along the optimum.  The cyclic state
boundary conditions make p_q, p_z periodic.

What this script does
---------------------
1. Works at parameters YOU choose (pass a ``params`` dict to ``main``), or, if
   none are given, scans for an M<0 balanced-growth point as before. The BG
   fixed point and the second-variation coefficients M, N are obtained from
   ``Second_variation_calculation.compute_MN`` at those parameters.
2. Solves the cyclic minimum-doubling-time problem DIRECTLY (single shooting +
   SLSQP) over a fine piecewise-constant control f_i in [0,1], multi-started
   from BG, smooth high-frequency seeds, and bang-bang (PWM) seeds.  Bang-bang
   and singular arcs emerge naturally as the f_i pin to 0/1 or stay interior.
3. Recovers the costates (p_q, p_z periodic, p_w constant, H == 0) along the
   optimal orbit and computes the switching function phi(t).
4. Classifies every instant as bang-0 / bang-1 / singular and CHECKS PMP
   consistency: f=1 where phi<0, f=0 where phi>0, phi~0 on interior arcs.
5. Compares the optimal doubling time to balanced growth and plots the control
   (arc-coloured), phi(t), the (q,z) limit cycle, and the doubling curve.

Output: ``mm_pmp_optimal.png``.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Declining_direction_search import (  # noqa: E402
    find_Mneg_point, smallest_negative_n, compute_MN,
    S_fun, F_fun, G_fun, lam_fun,
)

LN2 = np.log(2.0)


# ============================================================
# Dynamics and their (q,z) Jacobian (for the costate equations)
# ============================================================

def dynamics(q, z, f, pars):
    kR, kT, kA, kC, Km = pars
    return (F_fun(q, z, f, kR, kT, kA, kC, Km),
            G_fun(q, z, f, kR, kT, kA, kC, Km),
            lam_fun(q, z, f, kR, kT, kA, kC, Km))


def dyn_jac(q, z, f, pars, h=1e-6):
    """Return A = d(qdot,zdot)/d(q,z) (2x2) and g = d lambda/d(q,z) (2,)."""
    fqp, fzp, lqp = dynamics(q + h, z, f, pars)
    fqm, fzm, lqm = dynamics(q - h, z, f, pars)
    fqp2, fzp2, lzp = dynamics(q, z + h, f, pars)
    fqm2, fzm2, lzm = dynamics(q, z - h, f, pars)

    A = np.array([
        [(fqp - fqm) / (2 * h), (fqp2 - fqm2) / (2 * h)],
        [(fzp - fzm) / (2 * h), (fzp2 - fzm2) / (2 * h)],
    ])
    g = np.array([(lqp - lqm) / (2 * h), (lzp - lzm) / (2 * h)])
    return A, g


# ============================================================
# Direct cyclic single-shooting simulation (piecewise-constant control)
# ============================================================

def control_pc(theta, fvec):
    """Piecewise-constant control: theta in [0,1) -> fvec[ floor(theta*N) ]."""
    N = len(fvec)
    idx = int(theta * N)
    if idx < 0:
        idx = 0
    elif idx >= N:
        idx = N - 1
    return fvec[idx]


def simulate_cycle(fvec, q0, z0, P, pars, nsteps=240, dense=False):
    """RK4 integration of (q,z,R) over one cycle theta in [0,1] (t=P*theta)."""
    h = 1.0 / nsteps

    def deriv(theta, q, z, R):
        f = control_pc(theta, fvec)
        dq, dz, lam = dynamics(q, z, f, pars)
        return P * dq, P * dz, P * lam * R

    q, z, R = q0, z0, 1.0
    if dense:
        TH = np.empty(nsteps + 1); Q = np.empty(nsteps + 1)
        Z = np.empty(nsteps + 1); RR = np.empty(nsteps + 1)
        FF = np.empty(nsteps + 1)
        TH[0], Q[0], Z[0], RR[0] = 0.0, q, z, R
        FF[0] = control_pc(0.0, fvec)

    for i in range(nsteps):
        th = i * h
        k1 = deriv(th, q, z, R)
        k2 = deriv(th + 0.5 * h, q + 0.5 * h * k1[0], z + 0.5 * h * k1[1], R + 0.5 * h * k1[2])
        k3 = deriv(th + 0.5 * h, q + 0.5 * h * k2[0], z + 0.5 * h * k2[1], R + 0.5 * h * k2[2])
        k4 = deriv(th + h, q + h * k3[0], z + h * k3[1], R + h * k3[2])
        q += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        R += h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if dense:
            TH[i + 1] = (i + 1) * h; Q[i + 1] = q; Z[i + 1] = z
            RR[i + 1] = R
            FF[i + 1] = control_pc(min((i + 1) * h, 1 - 1e-12), fvec)

    if dense:
        return dict(theta=TH, t=TH * P, q=Q, z=Z, R=RR, T=Q * RR, A=Z * RR, f=FF)
    return q, z, R


# ============================================================
# Direct optimal-control solve (minimise doubling time P)
# ============================================================

def solve_direct(N, pars, fstar, bg, n_c, nsteps=240):
    """Minimise P over (f_1..f_N in [0,1], q0, z0, P) with cyclic+doubling cons."""
    qstar, zstar, taustar = bg
    margin = min(fstar, 1.0 - fstar)
    th_mid = (np.arange(N) + 0.5) / N

    def unpack(x):
        return np.asarray(x[:N]), x[N], x[N + 1], x[N + 2]

    def objective(x):
        return x[N + 2]

    def objective_jac(x):
        g = np.zeros(N + 3); g[N + 2] = 1.0
        return g

    def eq_constraint(x):
        fvec, q0, z0, P = unpack(x)
        qN, zN, RN = simulate_cycle(fvec, q0, z0, P, pars, nsteps=nsteps)
        return np.array([qN - q0, zN - z0, RN - 2.0])

    bounds = [(0.0, 1.0)] * N + [(1e-4, 1e3), (1e-4, 1e4), (1e-3, 5.0 * taustar)]
    cons = [{"type": "eq", "fun": eq_constraint}]

    # ---- multi-start seeds ----
    seeds = []
    # (0) balanced growth
    seeds.append(np.concatenate([np.full(N, fstar), [qstar, zstar, taustar]]))
    # (1,2) smooth high-frequency perturbations off BG (both signs)
    for s in (+1.0, -1.0):
        f0 = np.clip(fstar + s * 0.5 * margin * np.sin(n_c * np.pi * th_mid), 0, 1)
        seeds.append(np.concatenate([f0, [qstar, zstar, taustar]]))
    # (3,4) bang-bang PWM at the critical frequency, duty ~ f* (mean stays ~f*)
    for nfreq in (n_c, n_c + 2):
        phase = (nfreq * th_mid) % 1.0
        f0 = np.where(phase < fstar, 1.0, 0.0)
        seeds.append(np.concatenate([f0, [qstar, zstar, taustar]]))

    best = {"P": taustar, "x": seeds[0].copy(), "feas": 0.0, "seed": "BG"}
    names = ["BG", "smooth+", "smooth-", f"PWM(n_c)", f"PWM(n_c+2)"]

    for name, x0 in zip(names, seeds):
        try:
            res = minimize(objective, x0, method="SLSQP", jac=objective_jac,
                           bounds=bounds, constraints=cons,
                           options=dict(maxiter=200, ftol=1e-11))
        except Exception:
            continue
        fvec, q0, z0, P = unpack(res.x)
        qN, zN, RN = simulate_cycle(fvec, q0, z0, P, pars, nsteps=nsteps)
        feas = max(abs(qN - q0), abs(zN - z0), abs(RN - 2.0))
        if feas < 1e-5 and P < best["P"] - 1e-9:
            best = {"P": P, "x": res.x.copy(), "feas": feas, "seed": name}

    return best, N


# ============================================================
# Costate recovery + switching function (PMP verification)
# ============================================================

def compute_switching(traj, pars):
    """Recover periodic costates (p_q,p_z), constant p_w (with H==0) and phi(t).

    Costate ODE in real time:  pdot = -A(t)^T p - p_w g(t).
    Linear in p_w: write p(t) = p_w * ptil(t).  Periodic ptil is found from the
    monodromy of the homogeneous part; p_w then fixed by <H> = 0.
    """
    kR, kT, kA, kC, Km = pars
    th = traj["theta"]; q = traj["q"]; z = traj["z"]; f = traj["f"]
    P = traj["t"][-1]
    Nn = len(th) - 1
    dt = P / Nn

    # integrate fundamental solution Phi (2x2) and particular psi (2,) forward,
    # for the unit-p_w forcing:  d/dt[Phi] = -A^T Phi,  d/dt[psi] = -A^T psi - g
    Phi = np.eye(2)
    psi = np.zeros(2)

    def At(i):
        A, g = dyn_jac(q[i], z[i], f[i], pars)
        return A, g

    for i in range(Nn):
        A0, g0 = At(i)
        A1, g1 = At(i + 1)
        Am = 0.5 * (A0 + A1); gm = 0.5 * (g0 + g1)   # midpoint approx

        def rhs_Phi(M): return -A0.T @ M
        # Heun (RK2) for stability of the linear time-varying system
        kP1 = -A0.T @ Phi
        kP2 = -A1.T @ (Phi + dt * kP1)
        Phi = Phi + 0.5 * dt * (kP1 + kP2)

        kpsi1 = -A0.T @ psi - g0
        kpsi2 = -A1.T @ (psi + dt * kpsi1) - g1
        psi = psi + 0.5 * dt * (kpsi1 + kpsi2)

    # periodic condition: ptil(0) = Phi*ptil(0) + psi  ->  (I-Phi) ptil0 = psi
    try:
        ptil0 = np.linalg.solve(np.eye(2) - Phi, psi)
    except np.linalg.LinAlgError:
        return None

    # propagate ptil(t) forward and accumulate <H>/p_w = <ptil.xdot + lambda>
    ptil = ptil0.copy()
    PT = np.empty((Nn + 1, 2)); PT[0] = ptil
    bracket = np.empty(Nn + 1)   # ptil . xdot + lambda  (so H = 1 + p_w*bracket)
    for i in range(Nn + 1):
        dq, dz, lam = dynamics(q[i], z[i], f[i], pars)
        bracket[i] = ptil[0] * dq + ptil[1] * dz + lam
        PT[i] = ptil
        if i < Nn:
            A0, g0 = At(i); A1, g1 = At(i + 1)
            k1 = -A0.T @ ptil - g0
            k2 = -A1.T @ (ptil + dt * k1) - g1
            ptil = ptil + 0.5 * dt * (k1 + k2)

    mean_bracket = np.mean(bracket)
    if abs(mean_bracket) < 1e-12:
        return None
    p_w = -1.0 / mean_bracket          # enforce <H> = 1 + p_w<bracket> = 0

    # switching function phi = S(-p_q B - p_z kR z + p_w kR), p = p_w*ptil
    phi = np.empty(Nn + 1)
    for i in range(Nn + 1):
        S = S_fun(q[i], z[i], kR, kT, kA, kC, Km)
        B = kT + kR * q[i]
        pq, pz = p_w * PT[i, 0], p_w * PT[i, 1]
        phi[i] = S * (-pq * B - pz * kR * z[i] + p_w * kR)

    return dict(phi=phi, p_w=p_w, p=p_w * PT, H_offset=mean_bracket)


def classify_arcs(f, phi, ftol=1e-3, phitol=None):
    """Label each instant bang0 / bang1 / singular and score PMP consistency."""
    if phitol is None:
        phitol = 0.02 * (np.max(np.abs(phi)) + 1e-30)
    label = np.empty(len(f), dtype=object)
    consistent = np.zeros(len(f), dtype=bool)
    for i in range(len(f)):
        if f[i] <= ftol:
            label[i] = "bang0"; consistent[i] = phi[i] >= -phitol
        elif f[i] >= 1.0 - ftol:
            label[i] = "bang1"; consistent[i] = phi[i] <= phitol
        else:
            label[i] = "singular"; consistent[i] = abs(phi[i]) <= 3 * phitol
    return label, consistent, phitol


# ============================================================
# Plot
# ============================================================

def make_figure(traj, sw, info, outdir):
    taustar = info["taustar"]; P = info["P"]; fstar = info["fstar"]
    qstar = info["qstar"]; zstar = info["zstar"]
    th = traj["theta"]; f = traj["f"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    # ---- A: optimal control, arc-coloured ----
    ax = axes[0, 0]
    if sw is not None:
        label = info["label"]
        cmap = {"bang0": "C0", "bang1": "C3", "singular": "C2"}
        pts = np.array([th, f]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = [cmap[label[i]] for i in range(len(f) - 1)]
        lc = LineCollection(segs, colors=cols, linewidths=2)
        ax.add_collection(lc)
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
        for lab, c in cmap.items():
            ax.plot([], [], color=c, lw=2, label=lab)
    else:
        ax.plot(th, f, "C3", lw=2, label="optimal f")
    ax.axhline(fstar, color="k", ls="--", lw=1.2, label=rf"BG $f^*$={fstar:.3f}")
    ax.set_xlabel(r"phase $\theta=t/P$"); ax.set_ylabel(r"control $f(\theta)$")
    ax.set_title("A. PMP-optimal control (bang-bang / singular arcs)")
    ax.legend(fontsize=9, loc="best")

    # ---- B: switching function ----
    ax = axes[0, 1]
    if sw is not None:
        ax.plot(th, sw["phi"], "C4", lw=2)
        ax.axhline(0.0, color="k", lw=1)
        ax.fill_between(th, sw["phi"], 0, where=sw["phi"] < 0,
                        color="C3", alpha=0.15, label=r"$\phi<0\to f=1$")
        ax.fill_between(th, sw["phi"], 0, where=sw["phi"] > 0,
                        color="C0", alpha=0.15, label=r"$\phi>0\to f=0$")
        ax.set_title(f"B. Switching function $\\phi(\\theta)$  "
                     f"(PMP consistent: {info['pmp_pct']:.0f}%)")
        ax.legend(fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "costate recovery failed", ha="center")
        ax.set_title("B. Switching function")
    ax.set_xlabel(r"phase $\theta$"); ax.set_ylabel(r"$\phi(\theta)$")

    # ---- C: (q,z) limit cycle ----
    ax = axes[1, 0]
    ax.plot(traj["q"], traj["z"], "C3", lw=2, label="optimal cycle")
    ax.plot([qstar], [zstar], "ko", ms=9, label="BG fixed point")
    ax.set_xlabel("q = T/R"); ax.set_ylabel("z = A/R")
    ax.set_title("C. Optimal orbit in ratio space")
    ax.legend(fontsize=9, loc="best")

    # ---- D: doubling curve + summary ----
    ax = axes[1, 1]
    t_bg = np.linspace(0.0, taustar, 400)
    ax.plot(t_bg, np.exp(LN2 / taustar * t_bg), "k", lw=2, label=r"BG $\tau^*$")
    ax.plot(traj["t"], traj["R"], "C3", lw=2, label=f"optimal P={P:.4f}")
    ax.axhline(2.0, color="gray", ls="--", lw=1)
    ax.axvline(taustar, color="k", ls=":", lw=1)
    ax.axvline(P, color="C3", ls=":", lw=1)
    ax.set_xlabel("time t"); ax.set_ylabel(r"$R(t)/R(0)$")
    dpc = 100.0 * (P - taustar) / taustar
    ax.set_title(rf"D. Doubling: $\tau^*$={taustar:.4f}, $P$={P:.4f} ({dpc:+.3f}%)")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        rf"Pontryagin-optimal cyclic strategy  |  $k_T$={info['kT']:.3f}, "
        rf"$k_A$={info['kA']:.3f}, $K_m$={info['Km']:g}, seed={info['seed']}",
        fontsize=13,
    )
    png = os.path.join(outdir, "mm_pmp_optimal.png")
    fig.savefig(png, dpi=150)
    print(f"\nSaved figure to: {png}")
    plt.show()


# ============================================================
# Build a balanced-growth point (+ M, N) from chosen parameters
# ============================================================

def setup_from_params(kR, kT, kA, kC, Km, guess=(1.0, 1.0)):
    """Solve BG and the second-variation coefficients at the GIVEN parameters.

    Returns the same record shape as ``find_Mneg_point`` (keys kR..Km plus
    q, z, f, lambda, tau, M, N, ...), or None if BG has no valid fixed point
    for these rate constants.
    """
    out = compute_MN(kR, kT, kA, kC, Km, guess=guess)
    if out is None:
        return None
    return dict(kR=kR, kT=kT, kA=kA, kC=kC, Km=Km, **out)


# ============================================================
# Main
# ============================================================

def main(params=None, Nseg=None, nsteps_opt=240):
    """Run the PMP-optimal cyclic-strategy search.

    params : dict or None
        None  -> scan parameter space for a point with M<0 (slow, original
                 behaviour).
        dict  -> use your own rate constants, e.g.
                 {"kR": 1.0, "kT": 10.0, "kA": 0.4, "kC": 1.0, "Km": 0.05}.
                 Missing kR / kC default to 1.0.
    Nseg : int or None
        Number of piecewise-constant control intervals (None -> auto, ~6*n_c,
        at least 40).
    nsteps_opt : int
        RK4 steps per cycle used inside the optimiser (raise for very
        high-frequency optima).
    """
    outdir = os.path.dirname(os.path.abspath(__file__))

    if params is None:
        print("No parameters given - scanning for an M<0 balanced-growth point ...")
        pt = find_Mneg_point()
        if pt is None:
            print("No M<0 point found in the scanned region.")
            return
    else:
        kR = params.get("kR", 1.0); kC = params.get("kC", 1.0)
        kT = params["kT"]; kA = params["kA"]; Km = params["Km"]
        print(f"Using chosen parameters: kR={kR:g} kT={kT:g} kA={kA:g} "
              f"kC={kC:g} Km={Km:g}")
        pt = setup_from_params(kR, kT, kA, kC, Km)
        if pt is None:
            print("Balanced growth has no valid fixed point for these "
                  "parameters. Try different values.")
            return

    kR, kT, kA, kC, Km = pt["kR"], pt["kT"], pt["kA"], pt["kC"], pt["Km"]
    qstar, zstar, fstar = pt["q"], pt["z"], pt["f"]
    taustar, M, N_ = pt["tau"], pt["M"], pt["N"]
    pars = (kR, kT, kA, kC, Km)
    bg = (qstar, zstar, taustar)

    if M < 0:
        n_c = smallest_negative_n(M, N_, taustar)
        decl = f"critical mode n_c={n_c} (a declining direction exists)"
    else:
        n_c = 4   # no declining direction; n_c only seeds optimiser diversity
        decl = "M>=0: BG passes the 2nd-order test -> expect the optimum = BG"

    print(f"  point: kT={kT:.4f} kA={kA:.4f} Km={Km:g}")
    print(f"  BG: q*={qstar:.4f} z*={zstar:.4f} f*={fstar:.4f} tau*={taustar:.5f}")
    print(f"  M={M:.3e}  N={N_:.3e}  {decl}")

    # Direct optimal-control solve. Resolution ~ a few times the critical mode
    # so bang-bang / singular structure can form.
    if Nseg is None:
        Nseg = max(40, 6 * n_c)
    print(f"\nDirect optimal-control solve ({Nseg} control intervals) ...")
    best, Nseg = solve_direct(Nseg, pars, fstar, bg, n_c, nsteps=nsteps_opt)
    fvec = best["x"][:Nseg]; q0 = best["x"][Nseg]; z0 = best["x"][Nseg + 1]
    P = best["x"][Nseg + 2]
    print(f"  best seed: {best['seed']}")
    print(f"  optimal P = {P:.6f}  vs  tau* = {taustar:.6f}  "
          f"({100*(taustar-P)/taustar:+.3f}% faster)  feas={best['feas']:.1e}")

    traj = simulate_cycle(fvec, q0, z0, P, pars, nsteps=1200, dense=True)

    # PMP verification: costates + switching function
    print("\nRecovering costates and switching function (PMP) ...")
    sw = compute_switching(traj, pars)
    pmp_pct = np.nan
    label = None
    if sw is not None:
        label, consistent, phitol = classify_arcs(traj["f"], sw["phi"])
        pmp_pct = 100.0 * np.mean(consistent)
        nb0 = np.sum(label == "bang0"); nb1 = np.sum(label == "bang1")
        nsg = np.sum(label == "singular")
        ntot = len(label)
        print(f"  p_w = {sw['p_w']:.4e}   |H| offset ~ {abs(sw['H_offset']):.2e}")
        print(f"  arcs: bang0 {100*nb0/ntot:.0f}%, bang1 {100*nb1/ntot:.0f}%, "
              f"singular {100*nsg/ntot:.0f}%")
        print(f"  PMP sign-consistency: {pmp_pct:.1f}% of instants")
    else:
        print("  costate recovery failed (degenerate monodromy).")

    # validity
    print(f"\n  validity: f in [{traj['f'].min():.3f},{traj['f'].max():.3f}], "
          f"q-return={abs(traj['q'][-1]-traj['q'][0]):.1e}, "
          f"z-return={abs(traj['z'][-1]-traj['z'][0]):.1e}, "
          f"R(P)={traj['R'][-1]:.5f}")

    info = dict(taustar=taustar, P=P, fstar=fstar, qstar=qstar, zstar=zstar,
                kT=kT, kA=kA, Km=Km, seed=best["seed"], label=label,
                pmp_pct=pmp_pct)
    make_figure(traj, sw, info, outdir)


if __name__ == "__main__":
    # ======================================================================
    # CHOOSE PARAMETERS HERE
    # ----------------------------------------------------------------------
    # Edit this dict to run any rate constants you like (kR / kC default to 1):
    PARAMS = {"kR": 1.0, "kT": 37.0, "kA": 0.4, "kC": 1.0, "Km": 0.1}
    #
    # Set PARAMS = None instead to auto-scan for a point with M<0 (slower).
    # Optional: override the control resolution / integration steps, e.g.
    #   main(params=PARAMS, Nseg=80, nsteps_opt=400)
    # ======================================================================
    main(params=PARAMS)
