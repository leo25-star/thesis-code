"""
This code generates Figure 8 in the thesis, 
which shows the transporter-first doubling time 
as a function of the initial transporter-to-ribosome ratio T0/R0 (theta), 
for various values of kA/kC and kR/kT. 
It also compares the transporter-first doubling time to the balanced-growth doubling time, 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# Global settings
# -----------------------------
R0 = 1.0
A0 = 0.0

kT = 1.0
kC = 1.0

t_max = 1000
eps = 1e-10


# Parameter grids
kA_over_kC_values = [0.5, 1.0, 2.0, 4.0]
kR_over_kT_values = [0.5, 1.0, 2.0, 4.0]

theta_values = np.linspace(0.05, 4.0, 400)


# -----------------------------
# Classifier
# -----------------------------
def classify_phase_log(phase_log):
    if "stayed_limited_transport_phase" in phase_log:
        return "stays limited"

    if "reentered_AA_limited_ribo_phase" in phase_log:
        return "leaves and re-enters"
    if "left_AA_limited_transport_phase" in phase_log:
        return "leaves, no re-entry"
    if "immediate_ribosome_limited" in phase_log:
        return "leaves, no re-entry"

    return "other"


# -----------------------------
# Main simulation
# -----------------------------
def transporter_first_doubling_time(theta, kA_over_kC, kR_over_kT):
    """
    Transporter-first strategy.

    Phase 1:
        fT = 1, fR = 0, until T doubles.

    Phase 2:
        fR = 1, fT = 0, until R doubles.

    Regime 1: ribosome-limited, if kC*R <= kA*T
        dR = fR*kR*R
        dT = fT*kT*R
        dA = kA*T - kC*R

    Regime 2: amino-acid-limited, if A=0 and kA*T <= kC*R
        dR = fR*(kR*kA/kC)*T
        dT = fT*(kT*kA/kC)*T
        dA = 0
    """

    kA = kA_over_kC * kC
    kR = kR_over_kT * kT

    T0 = theta * R0

    R = R0
    T = T0
    A = A0
    t_total = 0.0
    phase_log = []

    # ============================================================
    # Phase 1: transporter-first, fT=1, fR=0
    # ============================================================

    # If starting AA-limited:
    if A <= eps and kA * T < kC * R:
        T_exit = kC * R / kA
        T_target = 2.0 * T0

        # AA-limited TF dynamics:
        # dT/dt = (kT*kA/kC)*T
        gamma_T = kT * kA / kC

        if T_target <= T_exit + eps:
            # T doubles before leaving AA limitation
            dt = np.log(T_target / T) / gamma_T
            t_total += dt
            T = T_target
            phase_log.append("stayed_limited_transport_phase")

        else:
            # Leave AA limitation before T doubles
            dt = np.log(T_exit / T) / gamma_T
            t_total += dt
            T = T_exit
            A = 0.0
            phase_log.append("left_AA_limited_transport_phase")

    else:
        phase_log.append("immediate_ribosome_limited")

    # If T has not doubled, continue in ribosome-limited regime
    if T < 2.0 * T0 - eps:

        def ode_transport_RL(t, y):
            T, A = y

            # R is fixed during TF phase
            R_fixed = R0

            dT = kT * R_fixed
            dA = kA * T - kC * R_fixed

            return [dT, dA]

        def event_T_doubled(t, y):
            return y[0] - 2.0 * T0

        event_T_doubled.terminal = True
        event_T_doubled.direction = 1

        def event_A_zero(t, y):
            return y[1]

        event_A_zero.terminal = True
        event_A_zero.direction = -1

        sol = solve_ivp(
            ode_transport_RL,
            [0, t_max],
            [T, A],
            events=[event_T_doubled, event_A_zero],
            rtol=1e-9,
            atol=1e-11,
            max_step=0.01,
        )

        if len(sol.t_events[0]) > 0:
            dt = sol.t_events[0][0]
            T, A = sol.y_events[0][0]
            t_total += dt
            phase_log.append("ribosome_limited_transport_phase")

        elif len(sol.t_events[1]) > 0:
            # This usually should not happen in TF after exit,
            # because T increases and R is fixed.
            dt = sol.t_events[1][0]
            T, A = sol.y_events[1][0]
            t_total += dt
            A = 0.0
            return np.nan, "unexpected_A_zero_during_transport"

        else:
            return np.nan, "failed_transport_RL"

    # ============================================================
    # Phase 2: ribosome-first after transporter doubling
    # ============================================================

    T_fixed = 2.0 * T0
    R = R0

    # If already AA-limited at start of ribosome phase
    if A <= eps and kA * T_fixed <= kC * R:
        phase_log.append("entered_AA_limited_ribo_phase")

    else:
        # Ribosome-limited ribosome phase
        def ode_ribo_RL(t, y):
            R, A = y

            dR = kR * R
            dA = kA * T_fixed - kC * R

            return [dR, dA]

        def event_R_doubled(t, y):
            return y[0] - 2.0 * R0

        event_R_doubled.terminal = True
        event_R_doubled.direction = 1

        def event_A_zero(t, y):
            return y[1]

        event_A_zero.terminal = True
        event_A_zero.direction = -1

        sol = solve_ivp(
            ode_ribo_RL,
            [0, t_max],
            [R, A],
            events=[event_R_doubled, event_A_zero],
            rtol=1e-9,
            atol=1e-11,
            max_step=0.01,
        )

        if len(sol.t_events[0]) > 0:
            dt = sol.t_events[0][0]
            t_total += dt
            phase_log.append("finished_before_reentry")
            return t_total, classify_phase_log(phase_log)

        elif len(sol.t_events[1]) > 0:
            dt = sol.t_events[1][0]
            R, A = sol.y_events[1][0]
            t_total += dt
            A = 0.0

            if kA * T_fixed <= kC * R + eps:
                phase_log.append("reentered_AA_limited_ribo_phase")
            else:
                return np.nan, "A_zero_but_not_AA_limited"

        else:
            return np.nan, "failed_ribo_RL"

    # ============================================================
    # AA-limited ribosome phase after entry/re-entry
    # ============================================================
    # fR=1, fT=0:
    # dR/dt = (kR*kA/kC)*T_fixed
    # T fixed
    # A=0
    # ============================================================

    dR_AA_limited = (kR * kA / kC) * T_fixed

    if dR_AA_limited <= 0:
        return np.nan, "failed_AA_limited_ribo"

    dt = (2.0 * R0 - R) / dR_AA_limited
    t_total += dt
    phase_log.append("finished_on_AA_limited_boundary")

    return t_total, classify_phase_log(phase_log)

# -----------------------------
# Run parameter sweep, plot figure 8
# -----------------------------
fig, axes = plt.subplots(
    len(kA_over_kC_values),
    len(kR_over_kT_values),
    figsize=(4.2 * len(kR_over_kT_values), 3.6 * len(kA_over_kC_values)),
    sharex=True,
    sharey=False,
)

summary = []
theta_min_grid = {}
tau_curve_grid = {}      # (i, j) -> TF doubling-time curve over theta_values
tau_bal_grid = {}        # (i, j) -> balanced-growth doubling time
tau_min_grid = {}        # (i, j) -> best (minimal) TF doubling time

for i, kA_over_kC in enumerate(kA_over_kC_values):
    for j, kR_over_kT in enumerate(kR_over_kT_values):

        ax = axes[i, j]

        kA = kA_over_kC * kC
        kR = kR_over_kT * kT

        # Balanced growth time
        tau_bal = (np.log(2.0) / kR) * (
            1.0 + (kC * kR) / (kA * kT)
        )

        ax.axhline(
            tau_bal,
            color="black",
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            label="balanced growth",
        )
        # Best TF time
        L = np.log(2)

        S = np.sqrt(
        kA * kT * L * (2 * kC * kR + kA * kT * L)
        )

        tau_TF = (
            kC / (kT * kA)
            * np.log(
                (2 * kC * kR)
                / (kC * kR - kA * kT * L + S)
            )
            + S / (kA * kR * kT)
        )

        taus = []

        for theta in theta_values:
            tau, clr = transporter_first_doubling_time(
                theta,
                kA_over_kC=kA_over_kC,
                kR_over_kT=kR_over_kT,
            )

            taus.append(tau)

        taus = np.array(taus)
        valid = np.isfinite(taus)

        tau_curve_grid[(i, j)] = taus
        tau_bal_grid[(i, j)] = tau_bal

        if np.any(valid):
            theta_min = theta_values[valid][np.argmin(taus[valid])]
            tau_min = np.min(taus[valid])
            theta_min_grid[(i, j)] = theta_min
            tau_min_grid[(i, j)] = tau_min

            summary.append(
                {
                    "kA/kC": kA_over_kC,
                    "kR/kT": kR_over_kT,
                    "theta_min": theta_min,
                    "tau_min": tau_min,
                    "tau_bal": tau_bal,
                }
            )

            
            ax.scatter(
                theta_values[valid],
                taus[valid],
                s=10,
                color = "green",
                alpha=0.8,
                label = "transporter-first"
                )

            ax.scatter(theta_min, tau_min, color="red", s=45, zorder=5, label="best TF")


        ax.set_title(
            rf"$k_A/k_C={kA_over_kC}$, $k_R/k_T={kR_over_kT}$",
            fontsize=10,
        )

        ax.grid(True, alpha=0.3)

        if i == len(kA_over_kC_values) - 1:
            ax.set_xlabel(r"$T_0/R_0$")

        if j == 0:
            ax.set_ylabel(r"$\tau$")

    # Zoom the y-axis around the BG line and the TF minimum so the two
                # doubling times are directly comparable (the TF curve rises out of
                # frame away from its minimum, which is fine here).
        ref = min(tau_bal, tau_min)
        ax.set_ylim(0.9 * ref, 1.6 * ref)
    


# Shared legend without duplicate labels
handles, labels = axes[0, 0].get_legend_handles_labels()
unique = dict(zip(labels, handles))

fig.legend(
    unique.values(),
    unique.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.992),
    ncol=4,
    frameon=False,
)

fig.suptitle(
    "Transporter-first vs balanced growth doubling time",
    fontsize=15,
    y=1.02,
)

plt.tight_layout()
plt.show()

# -----------------------------
# Print minima summary
# -----------------------------
print("\nSummary of minima:")
for row in summary:
    winner = "TF" if row["tau_min"] < row["tau_bal"] else "BG"

    print(
        f"kA/kC={row['kA/kC']:.3g}, "
        f"kR/kT={row['kR/kT']:.3g} | "
        f"best T0/R0={row['theta_min']:.4f}, "
        f"min TF tau={row['tau_min']:.4f}, "
        f"BG tau={row['tau_bal']:.4f}, "
        f"winner={winner}"
    )