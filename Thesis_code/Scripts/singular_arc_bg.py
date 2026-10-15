"""
singular_arc_bg.py
==================

Locate the balanced-growth (BG) singular arc of each smooth-limitation model
by imposing the Pontryagin determinant constraint on the BG manifold.

Reduced (dimensionless) ratio system, q = T/R, z = A/R, control f = f_R:

    q' = F = phi(q,z) [ (1 - f) - f q ]
    z' = G = q - phi(q,z) ( rho + f z ),        rho = v/u
    lambda = f phi(q,z)

On BG both vanish.  F = 0 fixes the allocation,

    f* = 1/(1 + q*)        i.e.    q* = (1 - f*)/f*,

which leaves a ONE-PARAMETER family of BG points indexed by f: for each f,
G = 0 is a quadratic in z and fixes z*(f).  The singular arc is the single
member of that family that PMP admits (see `constraint` below).

Each model is run SEPARATELY, with its own parameter set: see MODELS.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

# The three-panel figure is shrunk to \textwidth in the thesis PDF, so fonts
# and lines are set large enough to stay readable after that scaling.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 21,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.titlesize": 24,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})


# ============================================================
# Model parameter sets
# ============================================================
#
# These mirror MODELS in phase_transition_three_models.py, whose parameter
# tuple is P = (u, v, chi, Km, delta).  Two naming notes:
#
#   * The thesis writes phi with kappa = delta * K_m.  The `Km` slot of
#     phase_transition_three_models.py actually holds that kappa, so we call
#     it `kappa` here to avoid the ambiguity.  (For modMM that script always
#     uses delta = 1, where kappa and K_m coincide.)
#   * The amino-acid consumption constant in z' is rho = v/u, with
#     u = k_T/k_R and v = k_C/k_A.
#
# Which BG point?  phase_transition_three_models.py does NOT plot the coarse
# bracketing scan it uses to locate the transition -- those rows are internal.
# It draws its six figure panels at
#
#     swept value = (transition value) * MULTS,
#     MULTS = [0.55, 0.72, 0.88, 1.13, 1.4, 1.9]
#
# so its FIRST panel (top-left, before the transition) sits at 0.55 * the
# transition value.  The analytic transitions are
#
#     MM, conc :  u*   = 1 / chi
#     modMM    :  Km*  = 1 / (chi u - 1)
#
# We therefore pin each model's swept parameter to that first-panel value, so
# the singular arc found here is exactly the BG point of the first panel of
# that script's figure.  Set PANEL_MULT to another entry of MULTS to
# reproduce any of the other five panels.  `f_expected` is the f_R* that
# script reports at the same point, for cross-checking.

PANEL_MULT = 0.55          # MULTS[0] -> top-left panel
CHI = 1.5
V = 2.5

U_STAR = 1.0 / CHI                      # MM, conc transition:  u* = 1/chi
KAPPA_STAR = 1.0 / (CHI * 1.0 - 1.0)    # modMM transition at u = 1:  Km* = 2

MODELS = [
    dict(
        name="Standard MM model",
        kind="MM",
        u=PANEL_MULT * U_STAR,          # u* = 0.66667  ->  u = 0.36667
        v=V,
        chi=CHI,
        kappa=1.0,
        delta=0.0,                      # not used by the standard MM phi
        f_expected=0.188389,
    ),
    dict(
        name="Modified MM model",
        kind="modMM",
        u=1.0,
        v=V,
        chi=CHI,
        kappa=PANEL_MULT * KAPPA_STAR,  # Km* = 2.0  ->  kappa = 1.10
        delta=1.0,
        f_expected=0.373127,
    ),
    dict(
        name="Concentration limited model",
        kind="conc",
        u=PANEL_MULT * U_STAR,          # u* = 0.66667  ->  u = 0.36667
        v=V,
        chi=CHI,
        kappa=1.0,                      # not used by the concentration phi
        delta=0.0,                      # not used
        f_expected=0.088761,
    ),
]


def rho_of(p):
    """Amino-acid consumption constant of the z' equation."""
    return p["v"] / p["u"]


# ============================================================
# Saturation factor phi and its first derivatives
# ============================================================
#
#   MM      phi = z / [ z + kappa (1 + chi u q) ]
#   modMM   phi = z / [ z + delta + kappa (1 + chi u q) ]
#   conc    phi = z / ( 1 + chi u q )
#
# phi_q is what carries the chi*u dependence, and chi*u = 1 is the phase
# transition of the MM and concentration-limited models -- so it must not be
# dropped from the optimality condition.

def phi_C(kind, q, p):
    """The q-dependent part of the phi denominator, and its q-derivative."""
    chi_u = p["chi"] * p["u"]
    offset = p["delta"] if kind == "modMM" else 0.0
    return offset + p["kappa"] * (1.0 + chi_u * q), p["kappa"] * chi_u


def phi_derivs(kind, q, z, p):
    """Return (phi, dphi/dq, dphi/dz)."""
    chi_u = p["chi"] * p["u"]

    if kind == "conc":
        D = 1.0 + chi_u * q
        return z / D, -z * chi_u / D**2, 1.0 / D

    C, Cq = phi_C(kind, q, p)
    D = z + C
    return z / D, -z * Cq / D**2, C / D**2


# ============================================================
# The BG family
# ============================================================

def bg_q(f):
    """F = 0  =>  q* = (1 - f)/f."""
    return (1.0 - f) / f


def bg_z(kind, f, p):
    """G = 0 solved for z on the BG manifold, for a given f_R.

    MM / modMM :   f z^2 + (rho - q) z - q C = 0
    conc       :   f z^2 + rho z - q (1 + chi u q) = 0
    """
    q = bg_q(f)
    rho = rho_of(p)

    if kind == "conc":
        b = 1.0 + p["chi"] * p["u"] * q
        return (-rho + np.sqrt(rho**2 + 4.0 * f * q * b)) / (2.0 * f)

    C, _ = phi_C(kind, q, p)
    return (-(rho - q) + np.sqrt((rho - q)**2 + 4.0 * f * q * C)) / (2.0 * f)


# ============================================================
# PMP / determinant constraint
# ============================================================
#
# With H = lambda + p1*F + p2*G, a BG point is a PMP candidate only if a
# NONZERO adjoint (p1,p2) exists solving all three stationarity conditions
#
#       dH/dq = dH/dz = dH/df = 0,
#
# i.e. only if the 3x3 matrix of gradients w.r.t. (q,z,f) is singular:
#
#       det[ grad(lambda) , grad(F) , grad(G) ] = 0.
#
# Using lambda = f*phi, F = phi*[(1-f) - f*q], G = q - phi*(rho + f*z), and
# imposing F = 0 (so f = 1/(1+q)), that determinant factors as
#
#       det = phi * [ phi^2 f^2 + rho*f*phi*phi_z - f*phi*phi_q - phi_z ].
#
# Since phi > 0, the constraint is the bracket:
#
#       D = phi^2*f_R^2 + rho*f_R*phi*phi_z - f_R*phi*phi_q - phi_z = 0
#
# This is the same condition as `stationarity` in
# phase_transition_three_models.py, which equals det/phi.

def constraint(kind, f, p):
    q = bg_q(f)
    z = bg_z(kind, f, p)
    phi, phi_q, phi_z = phi_derivs(kind, q, z, p)
    rho = rho_of(p)

    return (
        phi**2 * f**2
        + rho * f * phi * phi_z
        - f * phi * phi_q
        - phi_z
    )


# ============================================================
# Root finding
# ============================================================

# Avoid exactly f_R = 0 and f_R = 1, because q = (1-f_R)/f_R diverges at 0.
F_GRID = np.linspace(1e-4, 1.0 - 1e-4, 40001)


def bracket_roots(f_grid, D):
    """Sign-change roots of D(f), refined by linear interpolation."""
    roots = []
    for i in np.where(D[:-1] * D[1:] < 0.0)[0]:
        f1, f2 = f_grid[i], f_grid[i + 1]
        d1, d2 = D[i], D[i + 1]
        roots.append(f1 - d1 * (f2 - f1) / (d2 - d1))
    return roots


def refine(kind, f0, p, iters=80):
    """Tighten a root by bisection on the constraint."""
    h = 2.0 * (F_GRID[1] - F_GRID[0])
    a, b = max(1e-9, f0 - h), min(1.0 - 1e-9, f0 + h)
    Da, Db = constraint(kind, a, p), constraint(kind, b, p)
    if Da * Db > 0.0:
        return f0
    for _ in range(iters):
        m = 0.5 * (a + b)
        Dm = constraint(kind, m, p)
        if Da * Dm <= 0.0:
            b, Db = m, Dm
        else:
            a, Da = m, Dm
    return 0.5 * (a + b)


# ============================================================
# Run each model separately
# ============================================================

def run_model(p, ax):
    kind, name = p["kind"], p["name"]
    rho = rho_of(p)

    print("=" * 70)
    print(f"{name}   [{kind}]")
    print(f"  u={p['u']:.6g}  v={p['v']:.6g}  chi={p['chi']:.6g}  "
          f"kappa={p['kappa']:.6g}  delta={p['delta']:.6g}")
    print(f"  rho = v/u = {rho:.6g}    chi*u = {p['chi']*p['u']:.6g}")

    with np.errstate(all="ignore"):
        D = constraint(kind, F_GRID, p)

    roots = [refine(kind, r, p) for r in bracket_roots(F_GRID, D)]

    if not roots:
        print("  no sign-changing root of the constraint on (0,1).")
    for f in roots:
        q = bg_q(f)
        z = bg_z(kind, f, p)
        phi, _, _ = phi_derivs(kind, q, z, p)
        lam = f * phi
        print(f"  singular arc:  f_R* = {f:.6f}")
        print(f"     q* = {q:.6f}   z* = {z:.6f}   phi* = {phi:.6f}")
        print(f"     lambda* = {lam:.6f}   tau* = {np.log(2.0)/lam:.6f}")
        print(f"     residual D(f_R*) = {constraint(kind, f, p):+.2e}")

    if p.get("f_expected") is not None and roots:
        err = abs(roots[0] - p["f_expected"])
        tag = "OK" if err < 1e-4 else "MISMATCH"
        print(f"  cross-check vs phase_transition_three_models.py: "
              f"f_R* = {p['f_expected']:.6f}  |diff| = {err:.2e}  [{tag}]")

    ax.plot(F_GRID, D, lw=2.5)
    ax.axhline(0.0, ls="--", lw=1.5, color="k")
    for f in roots:
        ax.axvline(f, ls=":", lw=2.0, color="C3")
        ax.plot([f], [0.0], "o", color="C3", ms=10)
    ax.set_xlabel(r"$f_R^{BG}$")
    ax.set_ylabel(r"$\det A\,/\,\phi$")
    ttl = name
    if roots:
        ttl += "\n" + rf"$f_R^*={roots[0]:.4f}$"
    ax.set_title(ttl)
    return roots


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 6.0))
    for prm, ax in zip(MODELS, axes):
        run_model(prm, ax)
    for ax in axes[1:]:
        ax.set_ylabel("")       # y label only on the leftmost panel
    fig.suptitle(r"Constraint on $f_R^{BG}$")
    fig.tight_layout()

    # Figures are written to Thesis_code/PNGs (one level up from Scripts/)
    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PNGs"
    )
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "singular_arc_bg.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"saved {png}")

    plt.show()
