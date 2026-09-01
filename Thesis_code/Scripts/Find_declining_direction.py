"""
Declining_direction_search.py
==============================

Goal
----
For the Michaelis--Menten allocation model (see ``main (1).tex``), the balanced
growth (BG) solution is a critical point of the doubling time.  Its local
optimality is governed by the second variation

        delta^2 tau[y] = \int_0^{tau*} ( M ydot^2 + N y^2 ) dt ,

over scalar perturbations y(t) obeying the endpoint conditions

        y(0) = y(tau*) = 0 ,    ydot(0) = ydot(tau*) = 0 .

When  M < 0  the form is unbounded below in the high-frequency direction:
because  \int ydot^2  grows like n^2 while  \int y^2  stays O(1), a perturbation
with a large enough mode number n makes  delta^2 tau < 0 .  BG is then NOT a
local minimum -- there is a nearby admissible strategy that doubles faster.

This script:
  1. Scans parameter space (using the machinery in
     ``Second_variation_calculation.py``) to find a set of rate constants where
     M < 0 at the BG fixed point.
  2. Builds the n-dependent admissible variation
            y_n(t) = sin^2(pi t / tau*) * sin(n pi t / tau*)
     and finds the smallest n for which delta^2 tau[y_n] < 0.
  3. Reconstructs the control perturbation eta(t) that produces this variation,
     scales it (amplitude eps) so that f(t) = f* + eps*eta stays in [0, 1].
  4. Computes the *actual* (fully nonlinear) doubling time of that strategy by
     solving the cyclic boundary-value problem
            q(P) = q(0),  z(P) = z(0),  R(P)/R(0) = 2 ,
     for the period P (= doubling time) and the periodic initial ratios.
  5. Compares P with the BG doubling time tau*, and plots the doubling time,
     the strategy f(t), and the trajectories R, T, A for both strategies.

The nonlinear BVP is the ground truth: if delta^2 tau < 0 the solver should
return P < tau*, i.e. the variation really does double faster than BG.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# Reuse the validated model + second-variation machinery.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Calculate_second_variation import (  # noqa: E402
    S_fun, F_fun, G_fun, lam_fun,
    first_order_coeffs, solve_bg, compute_MN,
)

LN2 = np.log(2.0)


# ============================================================
# 1. Find a parameter set with M < 0
# ============================================================

def find_Mneg_point():
    """Scan a region of parameter space and return a BG point with M < 0.

    The note shows that M tends to become negative for large k_T/k_R, so we
    sweep k_T high and pick the valid point with the *most negative* M (the
    strongest declining direction, hence the clearest demonstration).
    """
    kR = 1.0
    kC = 1.0

    best = None

    for Km in [0.001, 0.01, 0.05, 0.1]:
        guess = (1.0, 1.0)
        for kT in np.logspace(0.5, 2.2, 20):
            for kA in np.logspace(-1.0, 0.6, 16):
                out = compute_MN(kR, kT, kA, kC, Km, guess=guess)
                if out is None:
                    continue
                guess = out["guess"]

                M = out["M"]
                q, z, f, lam = out["q"], out["z"], out["f"], out["lambda"]

                # keep only well-behaved, non-degenerate BG fixed points
                if not np.isfinite(M) or M >= 0:
                    continue
                if not (0.05 < f < 0.95):
                    continue
                if not (1e-2 < q < 1e2 and 1e-2 < z < 1e3):
                    continue
                if lam <= 0.05:
                    continue
                if abs(out["Hf"]) > 1e-2:        # control stationarity must hold
                    continue

                params = dict(kR=kR, kT=kT, kA=kA, kC=kC, Km=Km)
                rec = {**params, **out}
                if best is None or M < best["M"]:
                    best = rec

    return best


# ============================================================
# 2. The n-dependent admissible variation y_n(theta), theta = t / P
# ============================================================

def y_shape(theta, n):
    """Return y, dy/dtheta, d^2y/dtheta^2 for

        y(theta) = sin^2(pi theta) * sin(n pi theta),   theta in [0, 1].

    This obeys y = y' = 0 at theta = 0, 1 (the double zero from sin^2 kills
    both y and its first derivative), so all four endpoint conditions hold.
    """
    pi = np.pi
    p = np.sin(pi * theta) ** 2
    dp = pi * np.sin(2 * pi * theta)
    ddp = 2 * pi**2 * np.cos(2 * pi * theta)

    s = np.sin(n * pi * theta)
    ds = n * pi * np.cos(n * pi * theta)
    dds = -(n * pi) ** 2 * np.sin(n * pi * theta)

    y = p * s
    dy = dp * s + p * ds
    ddy = ddp * s + 2 * dp * ds + p * dds
    return y, dy, ddy


def second_variation(M, N, tau, n, ngrid=20001):
    """delta^2 tau[y_n] = M \int ydot^2 dt + N \int y^2 dt over [0, tau].

    With theta = t/tau:  \int ydot^2 dt = (1/tau) \int y'(theta)^2 dtheta,
    and \int y^2 dt = tau \int y(theta)^2 dtheta.
    """
    theta = np.linspace(0.0, 1.0, ngrid)
    y, dy, _ = y_shape(theta, n)
    Iy = np.trapz(y**2, theta)
    Idy = np.trapz(dy**2, theta)
    return M * Idy / tau + N * tau * Iy


def smallest_negative_n(M, N, tau, nmax=60):
    """Smallest mode number n such that delta^2 tau[y_n] < 0."""
    for n in range(1, nmax + 1):
        if second_variation(M, N, tau, n) < 0.0:
            return n
    return nmax


# ============================================================
# 3. Control perturbation eta(theta) reproducing y_n at period P
# ============================================================

def eta_time(theta, P, n, coef):
    """Control perturbation eta(t) that generates the variation y_n.

    From the linearized dynamics (see the note):
        x      = (ydot + b y) / gamma
        eta    = -(xdot + S* C x) / (S* B)
    Substituting x(y) gives
        eta = -1/(S* B gamma) [ yddot + (b + S* C) ydot + S* C b y ].
    With theta = t/P the time derivatives are ydot = y'/P, yddot = y''/P^2.
    """
    Sstar = coef["S"]
    B = coef["B"]
    C = coef["C"]
    b = coef["b"]
    gamma = coef["gamma"]

    y, dy, ddy = y_shape(theta, n)
    ydot = dy / P
    yddot = ddy / P**2
    return -1.0 / (Sstar * B * gamma) * (
        yddot + (b + Sstar * C) * ydot + Sstar * C * b * y
    )


def choose_eps(P, n, coef, fstar, frac=0.7):
    """Amplitude eps so that f = f* + eps*eta stays inside [0, 1].

    eps uses a fraction `frac` of the available margin min(f*, 1 - f*).
    """
    theta = np.linspace(0.0, 1.0, 4001)
    eta = eta_time(theta, P, n, coef)
    eta_max = np.max(np.abs(eta))
    margin = min(fstar, 1.0 - fstar)
    return frac * margin / eta_max, eta_max


# ============================================================
# 4. Nonlinear doubling time of the perturbed cyclic strategy
# ============================================================

def make_control(eps, P, n, coef, fstar):
    """Return f(theta) for the perturbed strategy (clipped to [0, 1])."""
    def f_of_theta(theta):
        f = fstar + eps * eta_time(theta, P, n, coef)
        return np.clip(f, 0.0, 1.0)
    return f_of_theta


def integrate_cycle(q0, z0, P, eps, n, coef, fstar, pars, dense=False, npts=4001):
    """Integrate (q, z, R) over one cycle theta in [0, 1] (t = P*theta).

    Returns the endpoint state, and -- if dense -- the full trajectory.
    """
    kR, kT, kA, kC, Km = pars
    f_of_theta = make_control(eps, P, n, coef, fstar)

    def rhs(theta, Y):
        q, z, R = Y
        f = float(f_of_theta(theta))
        dq = P * F_fun(q, z, f, kR, kT, kA, kC, Km)
        dz = P * G_fun(q, z, f, kR, kT, kA, kC, Km)
        dR = P * lam_fun(q, z, f, kR, kT, kA, kC, Km) * R
        return [dq, dz, dR]

    t_eval = np.linspace(0.0, 1.0, npts) if dense else None
    sol = solve_ivp(
        rhs, (0.0, 1.0), [q0, z0, 1.0],
        method="LSODA", rtol=1e-10, atol=1e-12,
        t_eval=t_eval, dense_output=False,
    )
    return sol


def solve_perturbed_doubling(eps, n, coef, fstar, pars, bg_guess):
    """Solve the cyclic BVP for the perturbed strategy.

    Unknowns: periodic initial ratios (q0, z0) and the period P (= doubling
    time).  Conditions: q(P)=q0, z(P)=z0, R(P)/R(0)=2.
    """
    qstar, zstar, taustar = bg_guess

    def residual(u):
        q0, z0, P = np.exp(u)
        sol = integrate_cycle(q0, z0, P, eps, n, coef, fstar, pars)
        if not sol.success:
            return [1e3, 1e3, 1e3]
        qP, zP, RP = sol.y[:, -1]
        return [qP - q0, zP - z0, RP - 2.0]

    u0 = np.log([qstar, zstar, taustar])
    out = root(residual, u0, method="hybr", tol=1e-12)
    if not out.success:
        return None
    q0, z0, P = np.exp(out.x)
    return {"q0": q0, "z0": z0, "P": P, "success": out.success}


# ============================================================
# 5. Trajectory reconstruction for plotting
# ============================================================

def perturbed_trajectory(q0, z0, P, eps, n, coef, fstar, pars, npts=4001):
    sol = integrate_cycle(q0, z0, P, eps, n, coef, fstar, pars,
                          dense=True, npts=npts)
    theta = sol.t
    q, z, R = sol.y
    T = q * R
    A = z * R
    t = P * theta
    f = make_control(eps, P, n, coef, fstar)(theta)
    return dict(t=t, theta=theta, q=q, z=z, R=R, T=T, A=A, f=f)


def bg_trajectory(qstar, zstar, fstar, lam, taustar, npts=4001):
    t = np.linspace(0.0, taustar, npts)
    R = np.exp(lam * t)
    T = qstar * R
    A = zstar * R
    q = np.full_like(t, qstar)
    z = np.full_like(t, zstar)
    f = np.full_like(t, fstar)
    return dict(t=t, q=q, z=z, R=R, T=T, A=A, f=f)


# ============================================================
# 6. Plotting
# ============================================================

def make_plots(bg, pert, info, outdir):
    taustar = info["taustar"]
    P = info["P"]
    fstar = info["fstar"]
    n = info["n"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # ---- (0,0) strategy f(t) ----
    ax = axes[0, 0]
    ax.plot(bg["t"], bg["f"], color="k", lw=2, label=f"BG: $f^*={fstar:.4f}$")
    ax.plot(pert["t"], pert["f"], color="C3", lw=2,
            label=f"perturbed ($n={n}$)")
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("ribosome allocation $f(t)$")
    ax.set_title("Strategy")
    ax.legend()

    # ---- (0,1) doubling: growth factor R(t)/R(0) ----
    ax = axes[0, 1]
    ax.plot(bg["t"], bg["R"], color="k", lw=2, label="BG")
    ax.plot(pert["t"], pert["R"], color="C3", lw=2, label="perturbed")
    ax.axhline(2.0, color="gray", ls="--", lw=1)
    ax.axvline(taustar, color="k", ls=":", lw=1)
    ax.axvline(P, color="C3", ls=":", lw=1)
    ax.annotate(rf"$\tau^*={taustar:.4f}$", xy=(taustar, 1.05),
                color="k", ha="center")
    ax.annotate(rf"$P={P:.4f}$", xy=(P, 1.15), color="C3", ha="center")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"growth factor $R(t)/R(0)$")
    dpc = 100.0 * (P - taustar) / taustar
    ax.set_title(rf"Doubling time:  $P-\tau^* = {P-taustar:+.4e}$  ({dpc:+.3f}%)")
    ax.legend(loc="upper left")

    # ---- (1,0) R, T, A trajectories ----
    ax = axes[1, 0]
    ax.plot(bg["t"], bg["R"], color="C0", lw=2, ls="--", label="R (BG)")
    ax.plot(bg["t"], bg["T"], color="C1", lw=2, ls="--", label="T (BG)")
    ax.plot(bg["t"], bg["A"], color="C2", lw=2, ls="--", label="A (BG)")
    ax.plot(pert["t"], pert["R"], color="C0", lw=2, label="R (pert)")
    ax.plot(pert["t"], pert["T"], color="C1", lw=2, label="T (pert)")
    ax.plot(pert["t"], pert["A"], color="C2", lw=2, label="A (pert)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("molecule numbers (R(0)=1)")
    ax.set_title("Trajectories $R, T, A$")
    ax.legend(ncol=2, fontsize=9)

    # ---- (1,1) ratios q, z (show periodicity) ----
    ax = axes[1, 1]
    ax.plot(bg["t"], bg["q"], color="C4", lw=2, ls="--", label="q (BG)")
    ax.plot(bg["t"], bg["z"], color="C5", lw=2, ls="--", label="z (BG)")
    ax.plot(pert["t"], pert["q"], color="C4", lw=2, label="q (pert)")
    ax.plot(pert["t"], pert["z"], color="C5", lw=2, label="z (pert)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("ratios $q=T/R$, $z=A/R$")
    ax.set_title("Ratios return to start (cyclic strategy)")
    ax.legend(ncol=2, fontsize=9)

    sup = (rf"M < 0 declining direction  |  "
           rf"$M={info['M']:.3e}$, $N={info['N']:.3e}$, "
           rf"$\delta^2\tau={info['d2tau']:.3e}$")
    fig.suptitle(sup, fontsize=13)

    png = os.path.join(outdir, "declining_direction_search.png")
    fig.savefig(png, dpi=150)
    print(f"\nSaved figure to: {png}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    outdir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 64)
    print("1. Searching for a parameter set with M < 0 ...")
    print("=" * 64)
    pt = find_Mneg_point()
    if pt is None:
        print("No M<0 point found in the scanned region.")
        return

    kR, kT, kA, kC, Km = pt["kR"], pt["kT"], pt["kA"], pt["kC"], pt["Km"]
    qstar, zstar, fstar = pt["q"], pt["z"], pt["f"]
    lam, taustar = pt["lambda"], pt["tau"]
    M, N = pt["M"], pt["N"]

    print(f"  kR={kR:g}  kT={kT:.4f}  kA={kA:.4f}  kC={kC:g}  Km={Km:g}")
    print(f"  BG:  q*={qstar:.5f}  z*={zstar:.5f}  f*={fstar:.5f}")
    print(f"       lambda*={lam:.5f}  tau*={taustar:.5f}")
    print(f"       M={M:.5e}  N={N:.5e}   (M<0 -> not a local min)")

    # rebuild the first-order coefficients at this BG point
    coef = first_order_coeffs(qstar, zstar, kR, kT, kA, kC, Km)
    pars = (kR, kT, kA, kC, Km)

    print("\n" + "=" * 64)
    print("2. Finding the smallest mode n with delta^2 tau[y_n] < 0 ...")
    print("=" * 64)
    n = smallest_negative_n(M, N, taustar)
    d2tau = second_variation(M, N, taustar, n)
    print(f"  n = {n}   ->   delta^2 tau[y_n] = {d2tau:.6e}  (< 0)")
    # show the trend across n
    for nn in range(1, n + 2):
        print(f"     n={nn:2d}:  delta^2 tau = {second_variation(M, N, taustar, nn):+.6e}")

    print("\n" + "=" * 64)
    print("3. Building the admissible control perturbation ...")
    print("=" * 64)
    eps, eta_max = choose_eps(taustar, n, coef, fstar, frac=0.7)
    print(f"  eps = {eps:.6e}   (max|eta| = {eta_max:.4e} at P=tau*)")
    print(f"  control range: f in [{fstar - eps*eta_max:.4f}, "
          f"{fstar + eps*eta_max:.4f}]  (must stay in [0,1])")
    # predicted doubling-time change (perturbation theory)
    tau_pred = taustar + 0.5 * eps**2 * d2tau
    print(f"  predicted tau (2nd order): tau* + 1/2 eps^2 delta^2 tau "
          f"= {tau_pred:.6f}")

    print("\n" + "=" * 64)
    print("4. Solving the nonlinear cyclic BVP for the true doubling time ...")
    print("=" * 64)
    sol = solve_perturbed_doubling(eps, n, coef, fstar, pars,
                                   (qstar, zstar, taustar))
    if sol is None:
        print("  BVP solve failed.")
        return
    P = sol["P"]
    print(f"  periodic q0={sol['q0']:.6f}  z0={sol['z0']:.6f}")
    print(f"  perturbed doubling time P = {P:.6f}")
    print(f"  BG doubling time      tau* = {taustar:.6f}")
    print(f"  difference  P - tau*  = {P - taustar:+.6e}  "
          f"({100*(P-taustar)/taustar:+.4f} %)")
    if P < taustar:
        print("  => the variation DOUBLES FASTER than BG (BG is not optimal).")
    else:
        print("  => no speed-up found (check parameters / amplitude).")

    print("\n" + "=" * 64)
    print("5. Building trajectories and plotting ...")
    print("=" * 64)
    bg = bg_trajectory(qstar, zstar, fstar, lam, taustar)
    pert = perturbed_trajectory(sol["q0"], sol["z0"], P, eps, n, coef,
                                fstar, pars)

    info = dict(taustar=taustar, P=P, fstar=fstar, n=n,
                M=M, N=N, d2tau=d2tau)
    make_plots(bg, pert, info, outdir)


if __name__ == "__main__":
    main()
