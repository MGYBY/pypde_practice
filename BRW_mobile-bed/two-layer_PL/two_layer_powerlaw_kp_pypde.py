#!/usr/bin/env python3
"""Two-layer Karman-Pohlhausen model for power-law fluids.

The dimensionless state is

    Q = [h_l, h_u, q_l, q_u]

where h_i are layer thicknesses and q_i are layer discharges per unit
width.  The equations are supplied to PyPDE in the form

    Q_t + F(Q)_x + B(Q) Q_x = S(Q).

The closure reconstructs a local two-layer velocity profile by assuming
piecewise-linear shear-stress distributions.  This profile is exact for
the steady-uniform two-layer power-law flow, recovers the Ng-Mei
single-layer power-law profile as h_u -> 0, and recovers the quadratic
Newtonian two-layer reconstruction when n_l = n_u = 1.

Important model restriction
---------------------------
The present implementation follows the positive, monotone-shear branch

    tau_b > tau_I >= 0,  U_I > 0,  U_s - U_I >= 0.

It is intended for small-to-moderate perturbations of a downslope normal
flow.  Interfacial shear reversal and drying require an extended closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from numba import njit


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# --- Dimensionless physical parameters --------------------------------------
# Lower-layer Froude number based on steady-uniform lower mean velocity.
FR_LOWER = 0.50

# Steady-uniform depth and density ratios.
DEPTH_RATIO = 1.00                 # h_r = H_u / H_l
DENSITY_RATIO = 0.80               # r_rho = rho_u / rho_l

# Power-law indices (0 < n <= 1 for shear-thinning/Newtonian fluids).
N_LOWER = 0.40
N_UPPER = 0.80

# Scaled consistency ratio
#   kappa_K = (K_u/K_l) * (U_l/H_l)^(n_u-n_l) = Lambda_u/Lambda_l.
SCALED_CONSISTENCY_RATIO = 1.00

# --- Computational domain ---------------------------------------------------
DOMAIN_LENGTH = 2.0*(np.pi)/2.0
NX = 150                           # 200 cells per wavelength for 4 waves
FINAL_TIME = 6.0
N_OUTPUTS = 7

# --- Initial perturbation ---------------------------------------------------
PERTURBATION_TYPE = "periodic"     # "periodic" or "localized"
PERTURBATION_AMPLITUDE = 0.10
# PERTURBATION_WAVELENGTH = 5.0
PERTURBATION_WAVELENGTH = DOMAIN_LENGTH
UPPER_LAYER_PHASE_SHIFT = 0.0

# "local_normal_flow" uses the local gravitational tractions at the
# perturbed depths.  "froude_preserving" follows the h^(3/2) initialization
# used by Yu (2025) for the Newtonian integrated-layer model.
INITIAL_DISCHARGE_MODE = "local_normal_flow"

# --- PyPDE settings ----------------------------------------------------------
CFL = 0.880
RECONSTRUCTION_ORDER = 2
NUMERICAL_FLUX = "rusanov"         # "rusanov", "roe", or "osher"
STIFF_SOURCE = False
N_THREADS = 1
BOUNDARY_TYPE = "periodic"

# --- Output -----------------------------------------------------------------
OUTPUT_DIRECTORY = Path("two_layer_powerlaw_output")
OUTPUT_STEM = "two_layer_powerlaw_kp"
RUN_POSTPROCESSING = True

# --- Closure safeguards -----------------------------------------------------
MIN_DEPTH = 1.0e-8
MIN_VELOCITY = 1.0e-12
LAMBDA_MAX = 1.0 - 1.0e-11
LAMBDA_ITERATIONS = 52


# =============================================================================
# 2. PARAMETER VALIDATION AND DERIVED CONSTANTS
# =============================================================================

def _validate_parameters() -> None:
    if not (0.0 < N_LOWER <= 1.0 and 0.0 < N_UPPER <= 1.0):
        raise ValueError("Power-law indices must satisfy 0 < n <= 1.")
    if FR_LOWER <= 0.0:
        raise ValueError("FR_LOWER must be positive.")
    if DEPTH_RATIO <= 0.0:
        raise ValueError("DEPTH_RATIO must be positive.")
    if DENSITY_RATIO <= 0.0:
        raise ValueError("DENSITY_RATIO must be positive.")
    if SCALED_CONSISTENCY_RATIO <= 0.0:
        raise ValueError("SCALED_CONSISTENCY_RATIO must be positive.")
    if NX < 8:
        raise ValueError("NX must be at least 8.")
    if FINAL_TIME <= 0.0 or N_OUTPUTS < 1:
        raise ValueError("FINAL_TIME and N_OUTPUTS must be positive.")
    if PERTURBATION_WAVELENGTH <= 0.0:
        raise ValueError("PERTURBATION_WAVELENGTH must be positive.")


_validate_parameters()

M_LOWER = (N_LOWER + 1.0) / N_LOWER
M_UPPER = (N_UPPER + 1.0) / N_UPPER

# Upper-layer profile moments for f(r)=1-(1-r)^m.
UPPER_A = M_UPPER / (M_UPPER + 1.0)
UPPER_B = (
    1.0
    - 2.0 / (M_UPPER + 1.0)
    + 1.0 / (2.0 * M_UPPER + 1.0)
)


@njit(cache=True)
def lower_shape_coefficients(lam: float) -> Tuple[float, float, float]:
    """Return (C, A, B) for the lower-layer profile.

    For s=z/h_l and lambda=tau_I/tau_b,

        u_l/U_I = [1-(1-(1-lambda)s)^m] / (1-lambda^m),

    C relates U_I to basal stress,

        U_I = h_l (tau_b/Lambda_l)^(1/n_l) C,

    while A and B are the first two depth moments,

        q_l = h_l U_I A,
        M_l = h_l U_I^2 B.
    """
    m = M_LOWER

    if lam <= 1.0e-14:
        c = 1.0 / m
        a = m / (m + 1.0)
        b = 1.0 - 2.0 / (m + 1.0) + 1.0 / (2.0 * m + 1.0)
        return c, a, b

    eps = 1.0 - lam
    if eps < 1.0e-6:
        # Stable series for the nearly uniform-shear (Couette) limit.
        c = (
            1.0
            - 0.5 * (m - 1.0) * eps
            + (m - 1.0) * (m - 2.0) * eps * eps / 6.0
        )
        a = 0.5 + (m - 1.0) * eps / 12.0
        b = 1.0 / 3.0 + (m - 1.0) * eps / 12.0
        return c, a, b

    lam_m = lam ** m
    c = (1.0 - lam_m) / (m * (1.0 - lam))

    j_m = (1.0 - lam ** (m + 1.0)) / ((m + 1.0) * (1.0 - lam))
    j_2m = (
        (1.0 - lam ** (2.0 * m + 1.0))
        / ((2.0 * m + 1.0) * (1.0 - lam))
    )

    denom = 1.0 - lam_m
    a = (1.0 - j_m) / denom
    b = (1.0 - 2.0 * j_m + j_2m) / (denom * denom)
    return c, a, b


def _lower_shape_coefficients_python(lam: float) -> Tuple[float, float, float]:
    """Python wrapper used while constructing the global normal-flow state."""
    c, a, b = lower_shape_coefficients(float(lam))
    return float(c), float(a), float(b)


# Uniform traction ratio.  The stress distribution follows directly from
# the exact steady-uniform momentum balance.
UNIFORM_LAMBDA = (
    DENSITY_RATIO * DEPTH_RATIO
    / (1.0 + DENSITY_RATIO * DEPTH_RATIO)
)
UNIFORM_C, UNIFORM_A, UNIFORM_B = _lower_shape_coefficients_python(
    UNIFORM_LAMBDA
)

# Dimensionless power-law coefficients.  The lower coefficient is eliminated
# using the steady-uniform compatibility relation associated with the chosen
# lower-layer mean-velocity scale.
LAMBDA_LOWER = (
    (1.0 + DENSITY_RATIO * DEPTH_RATIO)
    * (UNIFORM_A * UNIFORM_C) ** N_LOWER
    / FR_LOWER**2
)
LAMBDA_UPPER = SCALED_CONSISTENCY_RATIO * LAMBDA_LOWER

UNIFORM_TAU_B = (
    1.0 + DENSITY_RATIO * DEPTH_RATIO
) / FR_LOWER**2
UNIFORM_TAU_I = (
    DENSITY_RATIO * DEPTH_RATIO
) / FR_LOWER**2

UNIFORM_U_INTERFACE = 1.0 / UNIFORM_A
UNIFORM_W_UPPER = (
    DEPTH_RATIO
    / M_UPPER
    * (UNIFORM_TAU_I / LAMBDA_UPPER) ** (1.0 / N_UPPER)
)
UNIFORM_U_UPPER = (
    UNIFORM_U_INTERFACE + UPPER_A * UNIFORM_W_UPPER
)
UNIFORM_Q_UPPER = DEPTH_RATIO * UNIFORM_U_UPPER


# =============================================================================
# 3. LOCAL KARMAN-POHLHAUSEN CLOSURE
# =============================================================================

@njit(cache=True)
def _closure_residual(
    lam: float,
    h_l: float,
    h_u: float,
    mean_l: float,
    mean_u: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """Residual of interfacial traction continuity on the positive branch."""
    c_l, a_l, b_l = lower_shape_coefficients(lam)

    u_interface = mean_l / a_l
    w_upper = (mean_u - u_interface) / UPPER_A

    velocity_scale_l = max(u_interface, MIN_VELOCITY) / (h_l * c_l)
    tau_b = LAMBDA_LOWER * velocity_scale_l**N_LOWER

    # The positive branch requires W >= 0.  The positive part gives a robust
    # bracket endpoint if the trial lambda makes W negative.
    shear_upper = M_UPPER * max(w_upper, 0.0) / h_u
    tau_i_upper = LAMBDA_UPPER * shear_upper**N_UPPER

    residual = lam * tau_b - tau_i_upper
    return residual, u_interface, w_upper, tau_b, c_l, a_l, b_l


@njit(cache=True)
def closure_terms(Q: np.ndarray) -> Tuple[
    float, float, float, float, float, float, float, float
]:
    """Recover stresses, profile speeds, and momentum moments.

    Returns
    -------
    lambda_i, tau_b, tau_i, U_I, W_u, M_l, M_u, branch_flag

    branch_flag = 1 means the positive monotone-shear branch was bracketed.
    branch_flag = 0 means an endpoint fallback was used.  The default initial
    conditions remain well inside the valid branch.
    """
    h_l = max(Q[0], MIN_DEPTH)
    h_u = max(Q[1], MIN_DEPTH)
    mean_l = Q[2] / h_l
    mean_u = Q[3] / h_u

    lo = 0.0
    hi = LAMBDA_MAX

    r_lo, _, _, _, _, _, _ = _closure_residual(
        lo, h_l, h_u, mean_l, mean_u
    )
    r_hi, _, _, _, _, _, _ = _closure_residual(
        hi, h_l, h_u, mean_l, mean_u
    )

    branch_flag = 1.0

    if r_lo >= 0.0:
        lam = lo
        branch_flag = 0.0
    elif r_hi <= 0.0:
        lam = hi
        branch_flag = 0.0
    else:
        for _ in range(LAMBDA_ITERATIONS):
            mid = 0.5 * (lo + hi)
            r_mid, _, _, _, _, _, _ = _closure_residual(
                mid, h_l, h_u, mean_l, mean_u
            )
            if r_mid > 0.0:
                hi = mid
            else:
                lo = mid
        lam = 0.5 * (lo + hi)

    _, u_interface, w_upper, tau_b, _, _, b_l = _closure_residual(
        lam, h_l, h_u, mean_l, mean_u
    )
    tau_i = lam * tau_b

    momentum_l = h_l * u_interface * u_interface * b_l
    momentum_u = h_u * (
        u_interface * u_interface
        + 2.0 * UPPER_A * u_interface * w_upper
        + UPPER_B * w_upper * w_upper
    )

    if u_interface <= 0.0 or w_upper < -1.0e-10:
        branch_flag = 0.0

    return (
        lam,
        tau_b,
        tau_i,
        u_interface,
        w_upper,
        momentum_l,
        momentum_u,
        branch_flag,
    )


# =============================================================================
# 4. PyPDE CALLBACKS
# =============================================================================

def F(Q: np.ndarray) -> np.ndarray:
    """Conservative flux vector."""
    flux = np.zeros(4)
    h_l = max(Q[0], MIN_DEPTH)
    h_u = max(Q[1], MIN_DEPTH)

    _, _, _, _, _, momentum_l, momentum_u, _ = closure_terms(Q)

    flux[0] = Q[2]
    flux[1] = Q[3]
    flux[2] = momentum_l + 0.5 * h_l * h_l / FR_LOWER**2
    flux[3] = momentum_u + 0.5 * h_u * h_u / FR_LOWER**2
    return flux


def B(Q: np.ndarray) -> np.ndarray:
    """Nonconservative hydrostatic coupling matrix."""
    matrix = np.zeros((4, 4))
    h_l = max(Q[0], MIN_DEPTH)
    h_u = max(Q[1], MIN_DEPTH)

    matrix[2, 1] = DENSITY_RATIO * h_l / FR_LOWER**2
    matrix[3, 0] = h_u / FR_LOWER**2
    return matrix


def S(Q: np.ndarray) -> np.ndarray:
    """Gravity and rheological traction source vector."""
    source = np.zeros(4)
    h_l = max(Q[0], MIN_DEPTH)
    h_u = max(Q[1], MIN_DEPTH)

    _, tau_b, tau_i, _, _, _, _, _ = closure_terms(Q)

    source[2] = h_l / FR_LOWER**2 + tau_i - tau_b
    source[3] = h_u / FR_LOWER**2 - tau_i / DENSITY_RATIO
    return source


# =============================================================================
# 5. INITIAL CONDITIONS AND RUNNER
# =============================================================================

def uniform_state() -> np.ndarray:
    """Return the exact dimensionless steady-uniform state."""
    return np.array(
        [1.0, DEPTH_RATIO, 1.0, UNIFORM_Q_UPPER],
        dtype=float,
    )


def local_normal_flow_fluxes(h_l: float, h_u: float) -> Tuple[float, float]:
    """Return local normal-flow discharges for prescribed local depths.

    The gravitational tractions are evaluated with the same dimensionless
    material coefficients as the reference state.  This is an initialization
    device, not an assertion that a spatially varying profile is an exact
    equilibrium.
    """
    h_l = max(float(h_l), MIN_DEPTH)
    h_u = max(float(h_u), MIN_DEPTH)

    tau_b = (h_l + DENSITY_RATIO * h_u) / FR_LOWER**2
    tau_i = DENSITY_RATIO * h_u / FR_LOWER**2
    lam = tau_i / tau_b

    c_l, a_l, _ = _lower_shape_coefficients_python(lam)
    u_interface = h_l * (tau_b / LAMBDA_LOWER) ** (1.0 / N_LOWER) * c_l
    w_upper = (
        h_u
        / M_UPPER
        * (tau_i / LAMBDA_UPPER) ** (1.0 / N_UPPER)
    )

    q_l = h_l * u_interface * a_l
    q_u = h_u * (u_interface + UPPER_A * w_upper)
    return q_l, q_u


def _perturbation(x: float) -> float:
    phase = 2.0 * np.pi * x / PERTURBATION_WAVELENGTH

    if PERTURBATION_TYPE == "periodic":
        return np.sin(phase)

    if PERTURBATION_TYPE == "localized":
        left = 0.5 * PERTURBATION_WAVELENGTH
        right = PERTURBATION_WAVELENGTH
        if left <= x <= right:
            return np.sin(phase)
        return 0.0

    raise ValueError(
        "PERTURBATION_TYPE must be 'periodic' or 'localized'."
    )


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Construct cell-centred initial data."""
    Q0 = np.zeros((x.size, 4), dtype=float)

    for i, xi in enumerate(x):
        disturbance_l = _perturbation(float(xi))
        disturbance_u = np.sin(
            2.0 * np.pi * float(xi) / PERTURBATION_WAVELENGTH
            + UPPER_LAYER_PHASE_SHIFT
        )
        if PERTURBATION_TYPE == "localized":
            left = 0.5 * PERTURBATION_WAVELENGTH
            right = PERTURBATION_WAVELENGTH
            if not (left <= xi <= right):
                disturbance_u = 0.0

        h_l = 1.0 + PERTURBATION_AMPLITUDE * disturbance_l
        h_u = DEPTH_RATIO * (
            1.0 + PERTURBATION_AMPLITUDE * disturbance_u
        )

        if INITIAL_DISCHARGE_MODE == "local_normal_flow":
            q_l, q_u = local_normal_flow_fluxes(h_l, h_u)
        elif INITIAL_DISCHARGE_MODE == "froude_preserving":
            q_l = h_l**1.5
            q_u = UNIFORM_Q_UPPER * (h_u / DEPTH_RATIO) ** 1.5
        else:
            raise ValueError(
                "INITIAL_DISCHARGE_MODE must be 'local_normal_flow' "
                "or 'froude_preserving'."
            )

        Q0[i] = (h_l, h_u, q_l, q_u)

    return Q0


def print_parameter_summary() -> None:
    """Print the dimensionless model and reference-state parameters."""
    Qn = uniform_state()
    closure = closure_terms(Qn)

    print("\nTwo-layer power-law Karman-Pohlhausen model")
    print("------------------------------------------------------------")
    print(f"Fr_l                         = {FR_LOWER:.10g}")
    print(f"h_r                          = {DEPTH_RATIO:.10g}")
    print(f"rho_r                        = {DENSITY_RATIO:.10g}")
    print(f"n_l, n_u                     = {N_LOWER:.10g}, {N_UPPER:.10g}")
    print(f"scaled consistency ratio     = {SCALED_CONSISTENCY_RATIO:.10g}")
    print(f"Lambda_l, Lambda_u           = {LAMBDA_LOWER:.10g}, {LAMBDA_UPPER:.10g}")
    print(f"normal-flow lambda           = {UNIFORM_LAMBDA:.10g}")
    print(f"normal-flow q_u              = {UNIFORM_Q_UPPER:.10g}")
    print(f"normal-flow U_I              = {UNIFORM_U_INTERFACE:.10g}")
    print(f"normal-flow U_u              = {UNIFORM_U_UPPER:.10g}")
    print(f"closure source residual norm = {np.linalg.norm(S(Qn)):.3e}")
    print(f"closure branch flag          = {closure[-1]:.0f}")
    print("------------------------------------------------------------\n")


def main() -> None:
    """Run the PyPDE calculation and optional postprocessing."""
    try:
        from pypde import pde_solver
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyPDE is not installed in this Python environment.  "
            "Use the Python 3.8 environment described in README.md."
        ) from exc

    dx = DOMAIN_LENGTH / NX
    x = (np.arange(NX, dtype=float) + 0.5) * dx
    Q0 = initial_condition(x)

    print_parameter_summary()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_x.npy", x)
    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_Q0.npy", Q0)

    out = pde_solver(
        Q0,
        FINAL_TIME,
        [DOMAIN_LENGTH],
        F=F,
        B=B,
        S=S,
        boundaryTypes=BOUNDARY_TYPE,
        cfl=CFL,
        order=RECONSTRUCTION_ORDER,
        ndt=N_OUTPUTS,
        flux=NUMERICAL_FLUX,
        stiff=STIFF_SOURCE,
        nThreads=N_THREADS,
    )

    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_out.npy", out)

    if RUN_POSTPROCESSING:
        from two_layer_powerlaw_postprocess import postprocess_solution

        postprocess_solution(
            x=x,
            Q0=Q0,
            out=out,
            final_time=FINAL_TIME,
            output_directory=OUTPUT_DIRECTORY,
            output_stem=OUTPUT_STEM,
        )


if __name__ == "__main__":
    main()
