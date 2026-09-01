import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# ============================================================
# Model functions
# ============================================================

def S_fun(q, z, kR, kT, kA, kC, Km):
    return z / (z + 1.0 + Km*(1.0 + q))

def F_fun(q, z, f, kR, kT, kA, kC, Km):
    S = S_fun(q, z, kR, kT, kA, kC, Km)
    return S * (kT*(1.0 - f) - kR*f*q)

def G_fun(q, z, f, kR, kT, kA, kC, Km):
    S = S_fun(q, z, kR, kT, kA, kC, Km)
    return kA*q - S*(kC + f*kR*z)

def lam_fun(q, z, f, kR, kT, kA, kC, Km):
    S = S_fun(q, z, kR, kT, kA, kC, Km)
    return f*kR*S


# ============================================================
# Analytic S derivatives
# ============================================================

def S_derivatives(q, z, Km):
    D = z + 1.0 + Km*(1.0 + q)
    S = z / D
    Sq = -Km*z / D**2
    Sz = (1.0 + Km*(1.0 + q)) / D**2
    return S, Sq, Sz


# ============================================================
# First-order coefficients and stationarity
# ============================================================

def first_order_coeffs(q, z, kR, kT, kA, kC, Km):
    S, Sq, Sz = S_derivatives(q, z, Km)

    B = kT + kR*q
    f = kT / B
    C = kR*f

    m = kR*z / B

    a = (
        kA
        - (kC + C*z)*Sq
        + (kR*z*S*C)/B
    )

    b = (
        (kC + C*z)*Sz
        + S*C
    )

    gamma = a - b*m

    alpha = kR/B - Sq/S
    beta = Sz/S

    stationarity = alpha*b - beta*a

    return {
        "S": S,
        "Sq": Sq,
        "Sz": Sz,
        "B": B,
        "f": f,
        "C": C,
        "m": m,
        "a": a,
        "b": b,
        "gamma": gamma,
        "alpha": alpha,
        "beta": beta,
        "stationarity": stationarity,
    }


# ============================================================
# Solve BG using F=0, G=0, stationarity=0
# But F=0 gives f = kT/(kT+kR q), so solve only q,z:
#   G(q,z,f*) = 0
#   alpha b - beta a = 0
# ============================================================

def solve_bg(kR, kT, kA, kC, Km, guess=(1.0, 1.0)):
    def equations(log_vars):
        q = np.exp(log_vars[0])
        z = np.exp(log_vars[1])

        c = first_order_coeffs(q, z, kR, kT, kA, kC, Km)
        f = c["f"]

        G = G_fun(q, z, f, kR, kT, kA, kC, Km)
        stat = c["stationarity"]

        return np.array([G, stat], dtype=float)

    sol = root(equations, np.log(guess), method="hybr")

    if not sol.success:
        return None

    q, z = np.exp(sol.x)
    c = first_order_coeffs(q, z, kR, kT, kA, kC, Km)
    f = c["f"]

    if not np.isfinite(q) or not np.isfinite(z) or not np.isfinite(f):
        return None
    if q <= 0 or z <= 0 or not (0 < f < 1):
        return None

    return q, z, f, c


# ============================================================
# Numerical derivatives with respect to X=(q,z,f)
# ============================================================

def grad_numeric(func, X, pars, h=1e-5):
    X = np.asarray(X, dtype=float)
    g = np.zeros(3)

    for i in range(3):
        step = h * max(1.0, abs(X[i]))
        xp = X.copy()
        xm = X.copy()
        xp[i] += step
        xm[i] -= step

        g[i] = (func(xp, pars) - func(xm, pars)) / (2.0*step)

    return g


def hess_numeric(func, X, pars, h=1e-4):
    X = np.asarray(X, dtype=float)
    H = np.zeros((3, 3))

    f0 = func(X, pars)

    steps = np.array([h*max(1.0, abs(x)) for x in X])

    for i in range(3):
        ei = np.zeros(3)
        ei[i] = steps[i]

        fp = func(X + ei, pars)
        fm = func(X - ei, pars)

        H[i, i] = (fp - 2.0*f0 + fm) / steps[i]**2

        for j in range(i+1, 3):
            ej = np.zeros(3)
            ej[j] = steps[j]

            fpp = func(X + ei + ej, pars)
            fpm = func(X + ei - ej, pars)
            fmp = func(X - ei + ej, pars)
            fmm = func(X - ei - ej, pars)

            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0*steps[i]*steps[j])
            H[j, i] = H[i, j]

    return H


def lam_wrapper(X, pars):
    q, z, f = X
    kR, kT, kA, kC, Km = pars
    return lam_fun(q, z, f, kR, kT, kA, kC, Km)

def F_wrapper(X, pars):
    q, z, f = X
    kR, kT, kA, kC, Km = pars
    return F_fun(q, z, f, kR, kT, kA, kC, Km)

def G_wrapper(X, pars):
    q, z, f = X
    kR, kT, kA, kC, Km = pars
    return G_fun(q, z, f, kR, kT, kA, kC, Km)


# ============================================================
# Compute corrected M,N
# ============================================================

def compute_MN(kR, kT, kA, kC, Km, guess=(1.0, 1.0)):
    bg = solve_bg(kR, kT, kA, kC, Km, guess=guess)

    if bg is None:
        return None

    q, z, f, c = bg
    X = np.array([q, z, f], dtype=float)
    pars = (kR, kT, kA, kC, Km)

    # First derivatives
    gl = grad_numeric(lam_wrapper, X, pars)
    gF = grad_numeric(F_wrapper, X, pars)
    gG = grad_numeric(G_wrapper, X, pars)

    # Solve adjoint equations:
    # lambda_q + p1 F_q + p2 G_q = 0
    # lambda_z + p1 F_z + p2 G_z = 0
    Aadj = np.array([
        [gF[0], gG[0]],
        [gF[1], gG[1]],
    ], dtype=float)

    rhs = -np.array([gl[0], gl[1]], dtype=float)

    try:
        p1, p2 = np.linalg.solve(Aadj, rhs)
    except np.linalg.LinAlgError:
        return None

    # Check control stationarity H_f=0
    Hf = gl[2] + p1*gF[2] + p2*gG[2]

    # Hessians
    Hl = hess_numeric(lam_wrapper, X, pars)
    HF = hess_numeric(F_wrapper, X, pars)
    HG = hess_numeric(G_wrapper, X, pars)

    HH = Hl + p1*HF + p2*HG

    Hqq = HH[0, 0]
    Hqz = HH[0, 1]
    Hzz = HH[1, 1]
    Hqf = HH[0, 2]
    Hzf = HH[1, 2]
    Hff = HH[2, 2]

    lam = lam_fun(q, z, f, kR, kT, kA, kC, Km)
    tau = np.log(2.0) / lam

    S = c["S"]
    B = c["B"]
    C = c["C"]
    m = c["m"]
    a = c["a"]
    b = c["b"]
    gamma = c["gamma"]

    D = S*B
    d = b + S*C
    e = S*C*b

    P = Hqf + m*Hzf
    Q = b*Hqf + a*Hzf

    A_ydot2 = Hqq + 2*m*Hqz + m**2*Hzz
    A_y2 = b**2*Hqq + 2*a*b*Hqz + a**2*Hzz

    M = -(
        A_ydot2
        - (2.0/D)*(P*d - Q)
    ) / (2.0*lam*gamma**2)

    N = -(
        A_y2
        - (2.0/D)*Q*e
    ) / (2.0*lam*gamma**2)

    criterion = N + (np.pi**2 / tau**2)*M

    return {
        "q": q,
        "z": z,
        "f": f,
        "lambda": lam,
        "tau": tau,
        "M": M,
        "N": N,
        "criterion": criterion,
        "p1": p1,
        "p2": p2,
        "Hf": Hf,
        "Hff": Hff,
        "guess": (q, z),
    }


# ============================================================
# Grid scan
# ============================================================

def scan_grid(grid_n=20):
    kR = 1.0
    kC = 1.0

    kA_ratios = np.logspace(-2, 3, grid_n)
    kT_ratios = np.logspace(-2, 3, grid_n)
    Km_values = [0.001, 0.01, 0.1, 1.0, 10.0, 20, 50, 100]

    results = {}

    for Km in Km_values:
        M_grid = np.full((grid_n, grid_n), np.nan)
        N_grid = np.full((grid_n, grid_n), np.nan)
        C_grid = np.full((grid_n, grid_n), np.nan)
        Hf_grid = np.full((grid_n, grid_n), np.nan)

        guess = (1.0, 1.0)

        for i, kA in enumerate(kA_ratios):
            for j, kT in enumerate(kT_ratios):
                out = compute_MN(kR, kT, kA, kC, Km, guess=guess)

                if out is None:
                    continue

                M_grid[i, j] = out["M"]
                N_grid[i, j] = out["N"]
                C_grid[i, j] = out["criterion"]
                Hf_grid[i, j] = out["Hf"]

                guess = out["guess"]

        results[Km] = {
            "M": M_grid,
            "N": N_grid,
            "criterion": C_grid,
            "Hf": Hf_grid,
        }

        print(f"Finished Km={Km}")

    return results, kA_ratios, kT_ratios, Km_values


# ============================================================
# Plotting
# ============================================================

def sign_grid(X, tol=1e-100):
    out = np.full_like(X, np.nan)
    out[X > tol] = 1
    out[X < -tol] = -1
    out[np.abs(X) <= tol] = 0
    return out


def plot_signs(results, kA_ratios, kT_ratios, Km_values, quantity="M"):
    fig, axes = plt.subplots(
        1, len(Km_values),
        figsize=(4*len(Km_values), 4.5),
        constrained_layout=True
    )

    if len(Km_values) == 1:
        axes = [axes]

    for ax, Km in zip(axes, Km_values):
        Z = sign_grid(results[Km][quantity])

        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                np.log10(kT_ratios[0]),
                np.log10(kT_ratios[-1]),
                np.log10(kA_ratios[0]),
                np.log10(kA_ratios[-1]),
            ],
            vmin=-1,
            vmax=1,
            cmap="bwr"
        )

        ax.set_title(rf"sign({quantity}), $K_m={Km}$")
        ax.set_xlabel(r"$\log_{10}(k_T/k_R)$")
        ax.set_ylabel(r"$\log_{10}(k_A/k_C)$")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["negative", "zero", "positive"])

    plt.show()


def plot_summary(results, kA_ratios, kT_ratios, Km_values):
    fig, axes = plt.subplots(
        1, len(Km_values),
        figsize=(4*len(Km_values), 4.5),
        constrained_layout=True
    )

    if len(Km_values) == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("viridis", 4)

    for ax, Km in zip(axes, Km_values):
        M = results[Km]["M"]
        crit = results[Km]["criterion"]

        summary = np.zeros_like(M)

        valid = np.isfinite(M) & np.isfinite(crit)
        Mpos = valid & (M > 0)
        critpos = Mpos & (crit >= 0)

        summary[valid & (M <= 0)] = 1
        summary[Mpos & (crit < 0)] = 2
        summary[critpos] = 3

        im = ax.imshow(
            summary,
            origin="lower",
            aspect="auto",
            extent=[
                np.log10(kT_ratios[0]),
                np.log10(kT_ratios[-1]),
                np.log10(kA_ratios[0]),
                np.log10(kA_ratios[-1]),
            ],
            vmin=0,
            vmax=3,
            cmap=cmap
        )

        ax.set_title(rf"$K_m={Km}$")
        ax.set_xlabel(r"$\log_{10}(k_T/k_R)$")
        ax.set_ylabel(r"$\log_{10}(k_A/k_C)$")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "no solution",
        r"$M\leq0$",
        r"$M>0$, criterion fails",
        r"$M>0$, BG local min",
    ])

    fig.suptitle("Corrected Hamiltonian second-variation test", fontsize=14)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Single test point
    test = compute_MN(
        kR=1.0,
        kT=10.0,
        kA=0.4,
        kC=1.0,
        Km=0.145344399333,
        guess=(1.0, 1.0),
    )

    print("\nSingle-point test:")
    if test is None:
        print("Failed.")
    else:
        for key in ["q", "z", "f", "lambda", "tau", "M", "N", "criterion", "Hf", "Hff"]:
            print(f"{key:>10s} = {test[key]:+.6e}")

    # Grid scan
    results, kA_ratios, kT_ratios, Km_values = scan_grid(grid_n=40)

    plot_signs(results, kA_ratios, kT_ratios, Km_values, quantity="M")
    plot_signs(results, kA_ratios, kT_ratios, Km_values, quantity="N")
    plot_signs(results, kA_ratios, kT_ratios, Km_values, quantity="criterion")
    plot_summary(results, kA_ratios, kT_ratios, Km_values)