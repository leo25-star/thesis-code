
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

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

def q_of_f(f):
    return (1 - f) / f


RHO_MM = 2.5/0.367
RHO_MODIFIED = 2.5
RHO_CONCENTRATION = 2.5/0.367


def z_mm(f):
    rho = RHO_MM
    chi = 1.5
    u = 0.367
    Km = 1
    delta = 1
    q = q_of_f(f)
    disc = (rho - q)**2 + 4*f*Km*q*(1 + chi*u*q)
    return (q - rho + np.sqrt(disc)) / (2*f)

def z_modified(f):
    rho = RHO_MODIFIED
    chi = 1.5
    u = 1
    Km = 1.1
    delta = 1
    q = q_of_f(f)
    C = delta + Km*(1 + chi*u*q)
    disc = (rho - q)**2 + 4*f*q*C
    return (q - rho + np.sqrt(disc)) / (2*f)

def z_concentration(f):
    rho = RHO_CONCENTRATION
    chi = 1.5
    u = 0.367
    q = q_of_f(f)
    disc = rho**2 + 4*f*q*(1 + chi*u*q)
    return (-rho + np.sqrt(disc)) / (2*f)

def tau_from_z(f, z, rho):
    # On BG: lambda = f*phi and G = 0 gives phi = q/(rho + f z),
    # so lambda = (1-f)/(rho + f z) and tau = ln2/lambda.
    return np.log(2) * (rho + f*z) / (1 - f)

models = {
    "Standard MM model": (z_mm, RHO_MM),
    "Modified MM model": (z_modified, RHO_MODIFIED),
    "Concentration limited model": (z_concentration, RHO_CONCENTRATION),
}

fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 6.0))

f_values = np.linspace(0.01, 0.9, 1500)

for ax, (name, (z_func, rho)) in zip(axes, models.items()):
    tau_values = tau_from_z(
        f_values, z_func(f_values), rho
    )
    ax.plot(f_values, tau_values, lw=2.5)

    result = minimize_scalar(
        lambda f: float(tau_from_z(f, z_func(f), rho)),
        bounds=(0.001, 0.999),
        method="bounded",
        options={"xatol": 1e-12},
    )

    f_opt = result.x
    tau_opt = result.fun

    ax.axvline(f_opt, ls=":", lw=2.0, color="C3")
    ax.plot([f_opt], [tau_opt], "o", color="C3", ms=10)
    ax.set_title(name + "\n" + rf"$f_R^*={f_opt:.4f}$")
    ax.set_xlabel(r"$f_R^{BG}$")
    ax.set_ylabel(r"$\tau$")

    print(
        f"{name}: f_R* = {f_opt:.6f}, "
        f"tau_min = {tau_opt:.6f}"
    )

for ax in axes[1:]:
    ax.set_ylabel("")       # y label only on the leftmost panel
fig.suptitle(r"Doubling time $\tau$ on the BG manifold")
fig.tight_layout()

# Figures are written to Thesis_code/PNGs (one level up from Scripts/)
outdir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PNGs"
)
os.makedirs(outdir, exist_ok=True)
png = os.path.join(outdir, "tau_fR.png")
fig.savefig(png, dpi=300, bbox_inches="tight")
print(f"saved {png}")

plt.show()
