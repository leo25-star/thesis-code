"""
Multi-model optimal control search for cellular growth resource allocation.

Supports two model variants:
  - Model 1: Amino-acid limited (AA) regime with scaled growth: dR = fR*(kR/kC)*J
  - Model 2: AA regime with transporter-limited growth: dR = fR*kR*T

Uses scipy.integrate.solve_ivp with event detection for smooth regime switching.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution


# ============================================================
# CONFIGURATION
# ============================================================

class ModelConfig:
    """Configuration for model variant and parameters."""

    def __init__(self, model_type="model1"):
        """
        Args:
            model_type: "model1" (kR/kC scaling) or "model2" (kR*T scaling)
        """
        self.model_type = model_type
        assert model_type in ["model1", "model2"], f"Unknown model_type: {model_type}"

    def __repr__(self):
        return f"ModelConfig({self.model_type})"


# ============================================================
# DYNAMICS: RHS FUNCTION
# ============================================================

def rhs_rl(R, T, A, fR, kR, kT, kA, kC):
    """
    Ribosome-Limited regime: A > 0.

    dR/dt = fR * kR * R
    dT/dt = (1-fR) * kT * R
    dA/dt = kA * T - kC * R
    """
    dR = fR * kR * R
    dT = (1.0 - fR) * kT * R
    dA = kA * T - kC * R
    return dR, dT, dA


def rhs_aa_model1(R, T, A, fR, kR, kT, kA, kC):
    """
    Amino-Acid Limited regime for Model 1: A ≈ 0.
    Growth limited by amino acid availability with (kR/kC) scaling.

    J = min(kC * R, kA * T)
    dR/dt = fR * (kR / kC) * J
    dT/dt = (1-fR) * (kT / kC) * J
    dA/dt = kA * T - J
    """
    J = min(kC * R, kA * T)
    dR = fR * (kR / kC) * J
    dT = (1.0 - fR) * (kT / kC) * J
    dA = kA * T - J
    return dR, dT, dA


def rhs_aa_model2(R, T, A, fR, kR, kT, kA, kC):
    """
    Amino-Acid Limited regime for Model 2: A ≈ 0.
    Growth limited by transporter production (no scaling).

    dR/dt = fR * kR * T
    dT/dt = (1-fR) * kT * T
    dA/dt = kA * T - J, where J = min(kC * R, kA * T)
    """
    J = min(kC * R, kA * T)
    dR = fR * kR * T
    dT = (1.0 - fR) * kT * T
    dA = kA * T - J
    return dR, dT, dA


def get_rhs_aa(model_type):
    """Return the appropriate AA regime RHS based on model type."""
    if model_type == "model1":
        return rhs_aa_model1
    elif model_type == "model2":
        return rhs_aa_model2
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# EVENT DETECTION
# ============================================================

def event_A_to_zero(t, y, *args):
    """
    Event: A crosses zero from above.
    Triggers when A transitions from positive to AA-limited regime.
    """
    return y[2] - 1e-10  # y[2] is A


event_A_to_zero.terminal = False
event_A_to_zero.direction = -1  # Only trigger when A decreases through zero


def event_growth_complete(t, y, R0, T0):
    """
    Event: Both R and T have doubled (cell cycle complete).
    Terminal event to stop integration when BOTH have reached their targets.

    y = [R, T, A]

    Returns max(target_R - R, target_T - T), which is:
    - Positive while either R or T hasn't doubled
    - Zero when the last one reaches its target (both doubled)
    - Negative when both exceed targets
    """
    target_R = 2.0 * R0
    target_T = 2.0 * T0
    return max(target_R - y[0], target_T - y[1])


event_growth_complete.terminal = True
event_growth_complete.direction = -1  # Trigger when both growth targets are met


# ============================================================
# REGIME DETECTION & SWITCHING
# ============================================================

class RegimeSwitcher:
    """Tracks which regime the system is in (RL or AA)."""

    def __init__(self, model_type="model1", threshold=1e-10):
        self.model_type = model_type
        self.threshold = threshold
        self.rhs_aa = get_rhs_aa(model_type)

    def get_rhs(self, R, T, A, fR, kR, kT, kA, kC):
        """
        Return (dR, dT, dA, regime_mode) based on current state.
        """
        if A > self.threshold:
            dR, dT, dA = rhs_rl(R, T, A, fR, kR, kT, kA, kC)
            mode = "RL"
        else:
            dR, dT, dA = self.rhs_aa(R, T, A, fR, kR, kT, kA, kC)
            mode = "AA"

        return dR, dT, dA, mode


# ============================================================
# ODE SYSTEM WRAPPER
# ============================================================

def make_ode_system(control_fun, control_params, kR, kT, kA, kC, model_type="model1"):
    """
    Create an ODE system for use with solve_ivp.

    Args:
        control_fun: function(t, *control_params) -> fR
        control_params: tuple of parameters for control_fun
        kR, kT, kA, kC: rate constants
        model_type: "model1" or "model2"

    Returns:
        A callable suitable for solve_ivp(fun, ...)
    """
    switcher = RegimeSwitcher(model_type=model_type)

    def system(t, y):
        R, T, A = y
        fR = control_fun(t, *control_params)
        fR = np.clip(fR, 0.0, 1.0)

        dR, dT, dA, _ = switcher.get_rhs(R, T, A, fR, kR, kT, kA, kC)
        return [dR, dT, dA]

    return system


# ============================================================
# CONTROL STRATEGIES
# ============================================================

def control_balanced(t, R0, T0, kR, kT):
    """Balanced growth: maintain optimal T/R ratio."""
    return kT / (kT + kR * (T0 / R0))


def control_ribofirst(t, t_switch):
    """Ribo-first: fR=1 until t_switch, then fR=0."""
    return 1.0 if t < t_switch else 0.0


def control_transportfirst(t, t_switch):
    """Transporter-first: fR=0 until t_switch, then fR=1."""
    return 0.0 if t < t_switch else 1.0


def control_nsegment(t, levels, switch_times):
    """n-segment piecewise constant control."""
    for i, s in enumerate(switch_times):
        if t < s:
            return levels[i]
    return levels[-1]


# ============================================================
# SIMULATION
# ============================================================

def simulate(control_fun,
             control_params,
             R0, T0, A0,
             kR, kT, kA, kC,
             model_type="model1",
             T_max=20.0,
             rtol=1e-8,
             atol=1e-10,
             dense_output=True):
    """
    Simulate the ODE system until both R and T double or T_max.

    Uses scipy.integrate.solve_ivp with event detection for regime switching.

    Args:
        control_fun: function(t, *control_params) -> fR(t)
        control_params: tuple of parameters for control_fun
        R0, T0, A0: initial conditions
        kR, kT, kA, kC: rate constants
        model_type: "model1" or "model2"
        T_max: maximum integration time
        rtol, atol: tolerance parameters
        dense_output: whether to return dense output solution

    Returns:
        dict with keys:
            - success_RT: whether R and T doubled (defines cell cycle completion)
            - success_A: whether A doubled (steady-state validity check)
            - tau: time when R and T both doubled (or T_max if they did not)
            - R_end, T_end, A_end: final values
            - target_A: 2*A0
            - t, R, T, A, u: trajectory arrays (if dense_output=True)
    """

    # Create ODE system
    ode_system = make_ode_system(control_fun, control_params, kR, kT, kA, kC, model_type)

    # Define events
    events = [event_A_to_zero]

    # Growth complete event with closure
    def event_gc(t, y):
        return event_growth_complete(t, y, R0, T0)
    event_gc.terminal = True
    event_gc.direction = -1
    events.append(event_gc)

    # Solve ODE
    y0 = [R0, T0, A0]
    sol = solve_ivp(
        ode_system,
        t_span=(0, T_max),
        y0=y0,
        events=events,
        dense_output=dense_output,
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=0.01,
    )

    # Extract results
    R_end = sol.y[0, -1]
    T_end = sol.y[1, -1]
    A_end = sol.y[2, -1]
    t_end = sol.t[-1]

    success_RT = (R_end >= 2.0 * R0) and (T_end >= 2.0 * T0)
    success_A = A_end >= 2.0 * A0
    target_A = 2.0 * A0

    result = {
        "success_RT": success_RT,
        "success_A": success_A,
        "tau": t_end,
        "R_end": R_end,
        "T_end": T_end,
        "A_end": A_end,
        "target_A": target_A,
    }

    # Add trajectory if requested
    if dense_output:
        # Evaluate solution on a fine grid
        t_eval = np.linspace(0, t_end, max(500, int(t_end / 0.01)))
        y_eval = sol.sol(t_eval)

        u_eval = np.array([control_fun(t, *control_params) for t in t_eval])
        u_eval = np.clip(u_eval, 0.0, 1.0)

        result.update({
            "t": t_eval,
            "R": y_eval[0, :],
            "T": y_eval[1, :],
            "A": y_eval[2, :],
            "u": u_eval,
            "sol": sol,  # Store full solution object for diagnostics
        })

    return result


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective_from_sim(sim, R0, T0, A0, T_max):
    """
    Compute objective from simulation result.

    Penalty structure:
      - Large penalty if R or T fail to double (cell cycle incomplete)
      - Otherwise tau is the base objective
      - Equal-magnitude penalty added to tau if A fails to double:
        a strategy where A does not double is not at steady state and
        cannot be a repeatable optimum, so it is treated as infeasible
    """
    if not sim["success_RT"]:
        short_R = max(0.0, 2.0 * R0 - sim["R_end"])
        short_T = max(0.0, 2.0 * T0 - sim["T_end"])
        return T_max + 1e4 * (1.0 + short_R**2 + short_T**2)

    val = sim["tau"]

    if not sim["success_A"]:
        short_A = max(0.0, 2.0 * A0 - sim["A_end"])
        val += 1e4 * (1.0 + short_A**2)

    return val


# ============================================================
# ANALYTICAL PREDICTIONS
# ============================================================

def analytical_balanced_growth(kR, kT, kA, kC):
    """
    Analytical doubling time for Balanced Growth (BG).

    In BG: q = T/R = k_C/k_A (constant ratio)
           fR_BG = kT / (kT + kR * k_C / k_A)

    Assumes:
    - System stays in RL regime (A > 0 throughout)
    - R and T grow exponentially: R(t) = R0 * exp(μ*t), T(t) = T0 * exp(μ*t)
    - Growth rate μ = fR_BG * kR
    - Doubling time τ = ln(2) / μ

    Note: This formula is identical for Model 1 and Model 2 in the RL regime.
          AA regime may alter the result, but this gives the RL-only prediction.

    Args:
        kR, kT, kA, kC: model parameters

    Returns:
        tuple: (fR_BG, tau_analytical, regime_assumption)
               regime_assumption is "RL only" (may be invalid if A hits zero)
    """
    # BG control value
    fR_BG = kT / (kT + kR * (kC / kA))

    # Growth rate in RL regime
    mu = fR_BG * kR

    # Doubling time
    tau_analytical = np.log(2) / mu

    return fR_BG, tau_analytical, "RL only"


def analytical_ribofirst_rl_only(kR, kT, kA, kC):
    v = kC / kA
    u = kT / kR
    v_u = v / u
    tau_rf_analytical = 1 + kR * np.sqrt(np.log(2)**2 + v_u + 0.5 * v_u**2)

    return tau_rf_analytical, "RL only"


# ============================================================
# COMPARISON TO THEORY
# ============================================================

def compare_to_theory(results_dict, case):
    """
    Compare numerical optimization results to analytical predictions.

    Args:
        results_dict: dict mapping strategy name → optimize_strategy() result
        case: parameter dict with kR, kT, kA, kC

    Returns:
        comparison_table (str): formatted comparison table
    """
    kR = case["kR"]
    kT = case["kT"]
    kA = case["kA"]
    kC = case["kC"]

    # Analytical predictions
    fR_bg_ana, tau_bg_ana, note_bg = analytical_balanced_growth(kR, kT, kA, kC)
    tau_rf_ana, note_rf = analytical_ribofirst_rl_only(kR, kT, kA, kC)

    # Numerical results (if available)
    tau_bg_num = results_dict.get("balanced", {}).get("sim", {}).get("tau", np.nan)
    tau_rf_num = results_dict.get("ribofirst", {}).get("sim", {}).get("tau", np.nan)
    tau_tf_num = results_dict.get("transportfirst", {}).get("sim", {}).get("tau", np.nan)
    tau_unrestr_num = results_dict.get("unrestricted", {}).get("sim", {}).get("tau", np.nan)

    # Build comparison table
    table = "\n" + "="*100 + "\n"
    table += "COMPARISON: ANALYTICAL vs NUMERICAL RESULTS\n"
    table += "="*100 + "\n"
    table += f"{'Strategy':<20} {'Analytical τ':<20} {'Numerical τ':<20} {'Difference':<15} {'Valid':<10}\n"
    table += "-"*100 + "\n"

    # BG comparison
    success_bg = results_dict.get("balanced", {}).get("sim", {}).get("success_A", False)
    if not np.isnan(tau_bg_num):
        diff_bg = abs(tau_bg_num - tau_bg_ana)
        pct_bg = 100 * diff_bg / tau_bg_ana if tau_bg_ana > 0 else 0
        table += f"{'Balanced Growth':<20} {tau_bg_ana:<20.6f} {tau_bg_num:<20.6f} {diff_bg:<15.6f} {'✓' if success_bg else '✗':<10}\n"
        table += f"{'':>20} fR={fR_bg_ana:.4f}, {note_bg} ({pct_bg:.2f}% diff)\n"
    else:
        table += f"{'Balanced Growth':<20} {tau_bg_ana:<20.6f} {'N/A':<20} {'N/A':<15} {'N/A':<10}\n"
        table += f"{'':>20} fR={fR_bg_ana:.4f}, {note_bg}\n"

    # RF comparison
    success_rf = results_dict.get("ribofirst", {}).get("sim", {}).get("success_A", False)
    if not np.isnan(tau_rf_num):
        diff_rf = abs(tau_rf_num - tau_rf_ana)
        pct_rf = 100 * diff_rf / tau_rf_ana if tau_rf_ana > 0 else 0
        table += f"{'Ribosome-First':<20} {tau_rf_ana:<20.6f} {tau_rf_num:<20.6f} {diff_rf:<15.6f} {'✓' if success_rf else '✗':<10}\n"
        table += f"{'':>20} {note_rf:<20} {f'({pct_rf:.2f}% diff)':<20}\n"
    else:
        table += f"{'Ribosome-First':<20} {tau_rf_ana:<20.6f} {'N/A':<20} {'N/A':<15} {'N/A':<10}\n"

    # TF (no analytical formula yet, just numerical)
    success_tf = results_dict.get("transportfirst", {}).get("sim", {}).get("success_A", False)
    if not np.isnan(tau_tf_num):
        table += f"{'Transporter-First':<20} {'(see main.tex)':<20} {tau_tf_num:<20.6f} {'–':<15} {'✓' if success_tf else '✗':<10}\n"
    else:
        table += f"{'Transporter-First':<20} {'(see main.tex)':<20} {'N/A':<20} {'N/A':<15} {'N/A':<10}\n"

    # Unrestricted (true optimum)
    if not np.isnan(tau_unrestr_num):
        table += f"{'Unrestricted (opt)':<20} {'(true optimum)':<20} {tau_unrestr_num:<20.6f} {'–':<15} {'✓':<10}\n"

    table += "="*100 + "\n"

    # Summary interpretation
    table += "\nINTERPRETATION:\n"
    table += "-"*100 + "\n"

    if not np.isnan(tau_bg_num) and not np.isnan(tau_unrestr_num):
        if tau_bg_num <= 1.01 * tau_unrestr_num:
            table += "✓ BG is optimal (or near-optimal). Analytical predictions are validated.\n"
        else:
            gap = 100 * (tau_bg_num - tau_unrestr_num) / tau_unrestr_num
            table += f"✗ BG is suboptimal by ~{gap:.1f}%. Another strategy is superior.\n"

    if not np.isnan(tau_rf_num) and success_rf:
        table += f"✓ RF achieves A doubling (valid strategy).\n"
    elif not np.isnan(tau_rf_num):
        table += f"✗ RF fails A-doubling constraint (invalid for this case).\n"

    table += "="*100 + "\n"

    return table


# ============================================================
# OPTIMIZATION HELPERS
# ============================================================

def unpack_full_x(x, n_segments):
    """Unpack optimization vector for n-segment control."""
    levels = np.array(x[:n_segments])
    switch_raw = np.array(x[n_segments:n_segments + n_segments - 1])
    switch_times = np.sort(switch_raw)
    T0 = x[-2]
    A0 = x[-1]
    return levels, switch_times, T0, A0


def full_bounds(n_segments, T_max, T0_bounds, A0_bounds):
    """Generate bounds for n-segment optimization."""
    b = []
    for _ in range(n_segments):
        b.append((0.0, 1.0))      # control levels
    for _ in range(n_segments - 1):
        b.append((0.0, T_max))    # switch times
    b.append(T0_bounds)
    b.append(A0_bounds)
    return b


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_strategy(
    strategy_name,
    case,
    model_type="model1",
    T_max=20.0,
    maxiter=100,
    popsize=30,
    seed=0,
    verbose=True,
):
    """
    Optimize a single strategy (balanced, ribo-first, transporter-first, or full n-segment).

    Args:
        strategy_name: "balanced", "ribofirst", "transportfirst", or "nsegment"
        case: dict with keys R0, kR, kT, kA, kC, T0_bounds, A0_bounds
        model_type: "model1" or "model2"
        T_max: max integration time
        maxiter: max DE iterations
        popsize: DE population size
        seed: random seed
        verbose: print progress

    Returns:
        dict with result, optimal params, and trajectory
    """

    R0 = case["R0"]
    kR = case["kR"]
    kT = case["kT"]
    kA = case["kA"]
    kC = case["kC"]
    T0_bounds = case.get("T0_bounds", (0.05, 10.0))
    A0_bounds = case.get("A0_bounds", (0.001, 10.0))

    eval_counter = {"n": 0, "best": np.inf}

    if strategy_name == "balanced":

        def objective(x):
            T0, A0 = x
            sim = simulate(
                control_fun=control_balanced,
                control_params=(R0, T0, kR, kT),
                R0=R0, T0=T0, A0=A0,
                kR=kR, kT=kT, kA=kA, kC=kC,
                model_type=model_type,
                T_max=T_max,
                dense_output=False,
            )
            val = objective_from_sim(sim, R0, T0, A0, T_max)

            eval_counter["n"] += 1
            if val < eval_counter["best"]:
                eval_counter["best"] = val
                if verbose:
                    print(f"  BAL: eval {eval_counter['n']:4d}, tau={val:.5f}, T0={T0:.4f}, A0={A0:.4f}")

            return val

        bounds = [T0_bounds, A0_bounds]

    elif strategy_name == "ribofirst":

        def objective(x):
            t_switch, T0, A0 = x
            sim = simulate(
                control_fun=control_ribofirst,
                control_params=(t_switch,),
                R0=R0, T0=T0, A0=A0,
                kR=kR, kT=kT, kA=kA, kC=kC,
                model_type=model_type,
                T_max=T_max,
                dense_output=False,
            )
            val = objective_from_sim(sim, R0, T0, A0, T_max)

            eval_counter["n"] += 1
            if val < eval_counter["best"]:
                eval_counter["best"] = val
                if verbose:
                    print(f"  RF:  eval {eval_counter['n']:4d}, tau={val:.5f}, t_switch={t_switch:.4f}")

            return val

        bounds = [(0.0, T_max), T0_bounds, A0_bounds]

    elif strategy_name == "transportfirst":

        def objective(x):
            t_switch, T0, A0 = x
            sim = simulate(
                control_fun=control_transportfirst,
                control_params=(t_switch,),
                R0=R0, T0=T0, A0=A0,
                kR=kR, kT=kT, kA=kA, kC=kC,
                model_type=model_type,
                T_max=T_max,
                dense_output=False,
            )
            val = objective_from_sim(sim, R0, T0, A0, T_max)

            eval_counter["n"] += 1
            if val < eval_counter["best"]:
                eval_counter["best"] = val
                if verbose:
                    print(f"  TF:  eval {eval_counter['n']:4d}, tau={val:.5f}, t_switch={t_switch:.4f}")

            return val

        bounds = [(0.0, T_max), T0_bounds, A0_bounds]

    elif strategy_name == "nsegment":
        n_segments = case.get("n_segments", 3)

        def objective(x):
            levels, switch_times, T0, A0 = unpack_full_x(x, n_segments)
            sim = simulate(
                control_fun=control_nsegment,
                control_params=(levels, switch_times),
                R0=R0, T0=T0, A0=A0,
                kR=kR, kT=kT, kA=kA, kC=kC,
                model_type=model_type,
                T_max=T_max,
                dense_output=False,
            )
            val = objective_from_sim(sim, R0, T0, A0, T_max)

            eval_counter["n"] += 1
            if val < eval_counter["best"]:
                eval_counter["best"] = val
                if verbose:
                    print(f"  NS:  eval {eval_counter['n']:4d}, tau={val:.5f}, levels={np.round(levels,3)}")

            return val

        bounds = full_bounds(n_segments, T_max, T0_bounds, A0_bounds)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Run optimization
    if verbose:
        print(f"\n{strategy_name.upper()} optimization")

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        seed=seed,
        disp=False,
        workers=1,
    )

    # Get trajectory with optimized parameters
    if strategy_name == "balanced":
        T0, A0 = result.x
        sim = simulate(
            control_fun=control_balanced,
            control_params=(R0, T0, kR, kT),
            R0=R0, T0=T0, A0=A0,
            kR=kR, kT=kT, kA=kA, kC=kC,
            model_type=model_type,
            T_max=T_max,
            dense_output=True,
        )
    elif strategy_name == "ribofirst":
        t_switch, T0, A0 = result.x
        sim = simulate(
            control_fun=control_ribofirst,
            control_params=(t_switch,),
            R0=R0, T0=T0, A0=A0,
            kR=kR, kT=kT, kA=kA, kC=kC,
            model_type=model_type,
            T_max=T_max,
            dense_output=True,
        )
    elif strategy_name == "transportfirst":
        t_switch, T0, A0 = result.x
        sim = simulate(
            control_fun=control_transportfirst,
            control_params=(t_switch,),
            R0=R0, T0=T0, A0=A0,
            kR=kR, kT=kT, kA=kA, kC=kC,
            model_type=model_type,
            T_max=T_max,
            dense_output=True,
        )
    elif strategy_name == "nsegment":
        levels, switch_times, T0, A0 = unpack_full_x(result.x, case.get("n_segments", 3))
        sim = simulate(
            control_fun=control_nsegment,
            control_params=(levels, switch_times),
            R0=R0, T0=T0, A0=A0,
            kR=kR, kT=kT, kA=kA, kC=kC,
            model_type=model_type,
            T_max=T_max,
            dense_output=True,
        )

    return {
        "strategy": strategy_name,
        "result": result,
        "sim": sim,
        "evals": eval_counter["n"],
    }


# ============================================================
# UNRESTRICTED OPTIMAL CONTROL
# ============================================================

def optimize_unrestricted(
    case,
    model_type="model1",
    n_segments=10,
    T_max=20.0,
    maxiter=150,
    popsize=50,
    seed=0,
    verbose=True,
):
    """
    Optimal control search over all possible piecewise-constant strategies.

    Uses n-segment control with many segments (default n=10) to approximate
    the true optimal strategy without imposing structure (BG, RF, TF).

    Args:
        case: dict with R0, kR, kT, kA, kC, T0_bounds, A0_bounds
        model_type: "model1" or "model2"
        n_segments: number of control segments (higher = more flexible)
        T_max: max integration time
        maxiter: max DE iterations
        popsize: DE population size
        seed: random seed
        verbose: print progress

    Returns:
        dict with result, optimal params, trajectory, and segment info
    """

    R0 = case["R0"]
    kR = case["kR"]
    kT = case["kT"]
    kA = case["kA"]
    kC = case["kC"]
    T0_bounds = case.get("T0_bounds", (0.05, 10.0))
    A0_bounds = case.get("A0_bounds", (0.001, 10.0))

    eval_counter = {"n": 0, "best": np.inf}

    def objective(x):
        levels, switch_times, T0, A0 = unpack_full_x(x, n_segments)
        sim = simulate(
            control_fun=control_nsegment,
            control_params=(levels, switch_times),
            R0=R0, T0=T0, A0=A0,
            kR=kR, kT=kT, kA=kA, kC=kC,
            model_type=model_type,
            T_max=T_max,
            dense_output=False,
        )
        val = objective_from_sim(sim, R0, T0, A0, T_max)

        eval_counter["n"] += 1
        if val < eval_counter["best"]:
            eval_counter["best"] = val
            if verbose:
                print(
                    f"  UNRESTRICTED: eval {eval_counter['n']:4d}, tau={val:.5f}, "
                    f"levels={np.round(levels, 3)[:5]}..."
                )

        return val

    bounds = full_bounds(n_segments, T_max, T0_bounds, A0_bounds)

    if verbose:
        print(f"\nUNRESTRICTED OPTIMAL CONTROL ({n_segments}-segment search)")

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        seed=seed,
        disp=False,
        workers=1,
    )

    # Get trajectory with optimized parameters
    levels, switch_times, T0, A0 = unpack_full_x(result.x, n_segments)
    sim = simulate(
        control_fun=control_nsegment,
        control_params=(levels, switch_times),
        R0=R0, T0=T0, A0=A0,
        kR=kR, kT=kT, kA=kA, kC=kC,
        model_type=model_type,
        T_max=T_max,
        dense_output=True,
    )

    return {
        "strategy": "unrestricted",
        "result": result,
        "sim": sim,
        "evals": eval_counter["n"],
        "levels": levels,
        "switch_times": switch_times,
        "n_segments": n_segments,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_strategy_comparison(
    results_dict,
    case_name="",
    figsize=(16, 12),
):
    """
    Plot comparison of optimal strategies.

    Creates a 2x2 figure with:
      Top-left: Control trajectories (fR(t)) for all strategies
      Top-right: Ribosome count (R) for all strategies
      Bottom-left: Transporter count (T) for all strategies
      Bottom-right: Amino acid pool (A) for all strategies

    Args:
        results_dict: dict mapping strategy name to optimize_strategy() result
                      Expected keys: "balanced", "ribofirst", "transportfirst",
                                     "unrestricted" (optional), "nsegment" (optional)
        case_name: title suffix (e.g., model name or parameter set)
        figsize: figure size

    Returns:
        fig, axes objects
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")

    colors = {
        "balanced": "C0",
        "ribofirst": "C1",
        "transportfirst": "C2",
        "unrestricted": "C3",
        "nsegment": "C4",
    }

    linestyles = {
        "balanced": "-",
        "ribofirst": "--",
        "transportfirst": "-.",
        "unrestricted": ":",
        "nsegment": "-",
    }

    # Extract data and sort by tau (for legend ordering)
    strategies_sorted = sorted(
        results_dict.keys(),
        key=lambda s: results_dict[s]["sim"]["tau"]
    )

    # ---- Panel 1: Control trajectories (fR) ----
    ax = axes[0, 0]
    for strategy in strategies_sorted:
        res = results_dict[strategy]
        sim = res["sim"]
        tau = sim["tau"]
        label = f"{strategy} (τ={tau:.4f})"
        ax.plot(
            sim["t"],
            sim["u"],
            label=label,
            color=colors.get(strategy, "gray"),
            linestyle=linestyles.get(strategy, "-"),
            linewidth=2,
        )
    ax.set_ylabel(r"Control $f_R(t)$", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_title(f"Control Strategy Comparison {case_name}", fontsize=13, fontweight="bold")

    # ---- Panel 2: Ribosome count (R) ----
    ax = axes[0, 1]
    for strategy in strategies_sorted:
        res = results_dict[strategy]
        sim = res["sim"]
        R0 = sim["R"][0]
        ax.plot(
            sim["t"],
            sim["R"],
            label=strategy,
            color=colors.get(strategy, "gray"),
            linestyle=linestyles.get(strategy, "-"),
            linewidth=2,
        )
    ax.axhline(2.0 * R0, color="black", linestyle=":", alpha=0.5, label="Target (2R₀)")
    ax.set_ylabel("Ribosome count $R(t)$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_title("Ribosome Dynamics", fontsize=13, fontweight="bold")

    # ---- Panel 3: Transporter count (T) ----
    ax = axes[1, 0]
    for strategy in strategies_sorted:
        res = results_dict[strategy]
        sim = res["sim"]
        T0 = sim["T"][0]
        ax.plot(
            sim["t"],
            sim["T"],
            label=strategy,
            color=colors.get(strategy, "gray"),
            linestyle=linestyles.get(strategy, "-"),
            linewidth=2,
        )
    ax.axhline(2.0 * T0, color="black", linestyle=":", alpha=0.5, label="Target (2T₀)")
    ax.set_ylabel("Transporter count $T(t)$", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_title("Transporter Dynamics", fontsize=13, fontweight="bold")

    # ---- Panel 4: Amino acid pool (A) ----
    ax = axes[1, 1]
    for strategy in strategies_sorted:
        res = results_dict[strategy]
        sim = res["sim"]
        ax.plot(
            sim["t"],
            sim["A"],
            label=strategy,
            color=colors.get(strategy, "gray"),
            linestyle=linestyles.get(strategy, "-"),
            linewidth=2,
        )
    ax.axhline(0.0, color="black", linestyle="-", alpha=0.3)
    ax.set_ylabel("Amino acid pool $A(t)$", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_title("Amino Acid Dynamics", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig, axes


def plot_theory_comparison(
    results_dict,
    case,
    model_type="",
    figsize=(14, 10),
):
    """
    Plot comparison between numerical optimization results and analytical predictions.

    Creates a 2×2 figure with:
      Top-left:     Bar chart of tau for all strategies with analytical overlays
      Top-right:    Balanced growth fR(t) vs the analytical fR* constant
      Bottom-left:  Amino acid pool A(t)/A0 for all strategies (checks A doubling)
      Bottom-right: Side-by-side bars: analytical vs numerical tau for BG and RF

    Analytical predictions come from main.tex:
      - BG:  fR* = kT/(kT + kR*kC/kA),  tau_BG = ln(2)/(fR* * kR)  [RL regime]
      - RF:  tau_RF = 1 + kR*sqrt(ln(2)**2 + v/u + 0.5*(v/u)**2),
             with v = kC/kA, u = kT/kR  [RL regime]

    Args:
        results_dict: dict mapping strategy name → optimize_strategy() result
        case: dict with kR, kT, kA, kC, R0
        model_type: string label for titles (e.g., "Model 1")
        figsize: figure size

    Returns:
        fig, axes objects
    """
    kR = case["kR"]
    kT = case["kT"]
    kA = case["kA"]
    kC = case["kC"]

    fR_bg_ana, tau_bg_ana, _ = analytical_balanced_growth(kR, kT, kA, kC)
    tau_rf_ana, _ = analytical_ribofirst_rl_only(kR, kT, kA, kC)

    strategy_labels = {
        "balanced": "Balanced",
        "ribofirst": "Ribo-First",
        "transportfirst": "Transport-First",
        "unrestricted": "Unrestricted",
        "nsegment": "N-Segment",
    }
    colors = {
        "balanced": "C0",
        "ribofirst": "C1",
        "transportfirst": "C2",
        "unrestricted": "C3",
        "nsegment": "C4",
    }

    strategy_order = [s for s in
                      ["balanced", "ribofirst", "transportfirst", "unrestricted", "nsegment"]
                      if s in results_dict]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ---- Panel 1: Bar chart of tau values with analytical overlays ----
    ax = axes[0, 0]
    labels = [strategy_labels.get(s, s) for s in strategy_order]
    tau_nums = [results_dict[s]["sim"]["tau"] for s in strategy_order]
    bar_colors = [colors.get(s, "gray") for s in strategy_order]

    bars = ax.bar(labels, tau_nums, color=bar_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)

    ax.axhline(tau_bg_ana, color="C0", linestyle="--", linewidth=2, alpha=0.8,
               label=f"BG analytical (RL): {tau_bg_ana:.4f}")
    ax.axhline(tau_rf_ana, color="C1", linestyle="--", linewidth=2, alpha=0.8,
               label=f"RF analytical (RL): {tau_rf_ana:.4f}")

    ax.set_ylabel(r"Doubling time $\tau$", fontsize=11)
    ax.set_title(f"Strategy τ vs Analytical Predictions {model_type}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    # ---- Panel 2: BG control trajectory vs analytical fR* ----
    ax = axes[0, 1]
    if "balanced" in results_dict:
        sim_bg = results_dict["balanced"]["sim"]
        tau_num = sim_bg["tau"]
        ax.plot(sim_bg["t"], sim_bg["u"], color="C0", linewidth=2.5,
                label=f"Numerical $f_R(t)$  (τ={tau_num:.4f})")
        ax.axhline(fR_bg_ana, color="red", linestyle="--", linewidth=2,
                   label=f"Analytical $f_R^*$ = {fR_bg_ana:.4f}")
        ax.axvline(tau_num, color="C0", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"Numerical τ = {tau_num:.4f}")
        ax.axvline(tau_bg_ana, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"Analytical τ = {tau_bg_ana:.4f} (RL)")

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(r"$f_R(t)$", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Balanced Growth: Numerical $f_R(t)$ vs Analytical $f_R^*$",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: A(t)/A0 — check whether A doubles ----
    ax = axes[1, 0]
    for strategy in strategy_order:
        sim = results_dict[strategy]["sim"]
        A0_val = sim["A"][0]
        if A0_val > 0:
            ax.plot(sim["t"], sim["A"] / A0_val,
                    label=f"{strategy_labels.get(strategy, strategy)} "
                          f"(end={sim['A'][-1]/A0_val:.2f}×A₀)",
                    color=colors.get(strategy, "gray"), linewidth=2)

    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8,
               label="Target: $2A_0$")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(r"$A(t)\,/\,A_0$", fontsize=11)
    ax.set_title("Amino Acid Pool (normalised) — Does A Double?",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Analytical vs numerical tau side-by-side ----
    ax = axes[1, 1]
    ana_entries = []
    if "balanced" in results_dict:
        sim = results_dict["balanced"]["sim"]
        ana_entries.append(("Balanced\nGrowth", tau_bg_ana, sim["tau"],
                            sim.get("success_A", False)))
    if "ribofirst" in results_dict:
        sim = results_dict["ribofirst"]["sim"]
        ana_entries.append(("Ribo-\nFirst", tau_rf_ana, sim["tau"],
                            sim.get("success_A", False)))

    if ana_entries:
        xlabels = [d[0] for d in ana_entries]
        taus_a = [d[1] for d in ana_entries]
        taus_n = [d[2] for d in ana_entries]
        successes = [d[3] for d in ana_entries]

        x = np.arange(len(xlabels))
        w = 0.35
        bars_a = ax.bar(x - w / 2, taus_a, w, label="Analytical (RL only)",
                        color="coral", alpha=0.8, edgecolor="black", linewidth=0.5)
        bars_n = ax.bar(x + w / 2, taus_n, w, label="Numerical",
                        color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars_a, fmt="%.4f", fontsize=8, padding=2)
        ax.bar_label(bars_n, fmt="%.4f", fontsize=8, padding=2)

        # Mark A-doubling success on each numerical bar
        for bar, ok in zip(bars_n, successes):
            marker, col = ("✓", "green") if ok else ("✗", "red")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02, marker,
                    ha="center", va="bottom", fontsize=13, color=col)

        # Percent difference annotation
        for i, (ta, tn) in enumerate(zip(taus_a, taus_n)):
            if ta > 0:
                pct = 100 * (tn - ta) / ta
                ax.text(i, min(ta, tn) * 0.5, f"{pct:+.1f}%",
                        ha="center", fontsize=9, color="darkred")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel(r"Doubling time $\tau$", fontsize=11)
        ax.set_title("Analytical vs Numerical τ\n(✓/✗ = A doubled)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Theory Comparison {model_type}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes


def print_strategy_summary(results_dict, model_type=""):
    """
    Print a summary table of all strategies.

    Args:
        results_dict: dict mapping strategy name to optimize_strategy() result
        model_type: optional string to print (e.g., "Model 1", "Model 2")
    """

    print("\n" + "="*80)
    print(f"STRATEGY COMPARISON SUMMARY {model_type}")
    print("="*80)
    print(f"{'Strategy':<20} {'τ (doubling time)':<20} {'Success':<12} {'Evaluations':<12}")
    print("-"*80)

    # Sort by tau for readability
    strategies_sorted = sorted(
        results_dict.items(),
        key=lambda x: x[1]["sim"]["tau"]
    )

    for strategy, res in strategies_sorted:
        sim = res["sim"]
        tau = sim["tau"]
        success = "✓" if sim["success_RT"] else "✗"
        evals = res.get("evals", "N/A")

        print(f"{strategy:<20} {tau:<20.6f} {success:<12} {evals:<12}")

    print("="*80)
    print(f"Best strategy: {strategies_sorted[0][0]} with τ={strategies_sorted[0][1]['sim']['tau']:.6f}")
    print("="*80 + "\n")


# ============================================================
# MAIN: Example usage
# ============================================================

if __name__ == "__main__":

    # Example parameter sets
    case_model1 = {
        "name": "Model 1 (kR/kC scaling)",
        "R0": 1.0,
        "kR": 1.0, "kT": 0.8, "kA": 0.6, "kC": 1.0,
        "T0_bounds": (0.05, 10.0),
        "A0_bounds": (0, 10.0),
        "seed": 42,
    }

    case_model2 = {
        "name": "Model 2 (kR*T scaling)",
        "R0": 1.0,
        "kR": 1.0, "kT": 0.8, "kA": 0.6, "kC": 1.0,
        "T0_bounds": (0.05, 10.0),
        "A0_bounds": (0, 10.0),
        "seed": 42,
    }

    # Test both models
    for case, model_type in [(case_model1, "model1"), (case_model2, "model2")]:
        print(f"\n{'='*70}")
        print(f"{case['name']}")
        print(f"{'='*70}")

        # Optimize all strategies
        results = {}

        # Standard strategies
        for strategy in ["balanced", "ribofirst", "transportfirst"]:
            results[strategy] = optimize_strategy(
                strategy,
                case,
                model_type=model_type,
                T_max=20.0,
                maxiter=50,
                popsize=15,
                seed=case["seed"],
                verbose=True,
            )

        # Full unrestricted optimal control
        results["unrestricted"] = optimize_unrestricted(
            case,
            model_type=model_type,
            n_segments=10,
            T_max=20.0,
            maxiter=500,
            popsize=25,
            seed=case["seed"],
            verbose=True,
        )

        # Print summary table
        print_strategy_summary(results, f"({model_type})")

        # Compare numerical results to analytical predictions
        comparison = compare_to_theory(results, case)
        print(comparison)

        # Plot strategy trajectories
        fig, ax = plot_strategy_comparison(
            results,
            case_name=f"({case['name']})",
            figsize=(16, 12),
        )
        plt.savefig(
            f"strategy_comparison_{model_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"✓ Saved plot: strategy_comparison_{model_type}.png")
        plt.show()

        # Plot theory comparison
        fig2, ax2 = plot_theory_comparison(
            results,
            case=case,
            model_type=f"({case['name']})",
            figsize=(14, 10),
        )
        plt.savefig(
            f"theory_comparison_{model_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"✓ Saved plot: theory_comparison_{model_type}.png")
        plt.show()
