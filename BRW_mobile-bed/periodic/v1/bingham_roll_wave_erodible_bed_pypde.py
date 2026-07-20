#!/usr/bin/env python3
"""PyPDE solver for Bingham roll waves over an erodible bed.

Dimensionless state
-------------------
    U = [h, q, u_p, b]

Governing form
--------------
    U_t + F(U)_x + B(U) U_x = S(U)

The code is intentionally divided into a short parameter block, model
closures, PyPDE callbacks, the initial condition, and the solver call.
Postprocessing is kept in ``bingham_roll_wave_postprocess.py``.
"""

from pathlib import Path
import math

import numpy as np
from numba import njit


# =============================================================================
# 1. Parameters to edit
# =============================================================================

# Liu-Mei hydrodynamics
ALPHA = 0.30                     # uniform plug-depth fraction / yield ratio
BETA = 27.0                      # U0^2 / [g H cos(theta)]
BED_SLOPE = 0.05                 # tan(theta)

# Rickenmann-Exner closure
MOBILE_BED = True               # set True to evolve the bed
BED_POROSITY = 0.40              # lambda_p
DENSITY_RATIO = 2.208333333333 # s = rho_s/rho_m (e.g. 2650/1200)
D_REL = 0.10                     # d_m/H (e.g. 0.010 m / 0.10 m)
D90_D30 = 1.379310344828         # d_90/d_30 (e.g. 0.012/0.0087)
THETA_CRIT = 0.047

# Periodic perturbation
WAVENUMBER = 1.20
PERTURBATION_AMPLITUDE = 1.0e-1
INITIAL_BED_AMPLITUDE = 0.0

# PyPDE controls
NX = 128
FINAL_TIME = 40.0
N_OUTPUTS = 40
CFL = 0.5
RECONSTRUCTION_ORDER = 2
NUMERICAL_FLUX = "rusanov"
STIFF_SOURCE = True
N_THREADS = 1

# Closure regularization.  These affect constitutive evaluations only; raw
# layer depths are retained in the output for diagnosis.
DEPTH_FLOOR = 1.0e-8
VELOCITY_FLOOR = 1.0e-10
LAYER_FLOOR = 1.0e-8
DELTA_H0 = 1.0e-2
DELTA_HP = 5.0e-3
SIGN_EPS = 1.0e-10
THETA_SMOOTH = 1.0e-10

# Output controls
OUTPUT_DIRECTORY = Path("bingham_roll_wave_output")
# Use an empty tuple to export every stored time frame.
# SNAPSHOT_TIMES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
SNAPSHOT_DT = 5.0
SNAPSHOT_TIMES = np.arange(
    0.0,
    FINAL_TIME + 0.5 * SNAPSHOT_DT,
    SNAPSHOT_DT,
    dtype=float,
)
SAVE_TEXT = True
SAVE_PLOTS = False
SHOW_PLOTS = False
PLOT_DPI = 220


THETA = math.atan(BED_SLOPE)
SIN_THETA = math.sin(THETA)
COS_THETA = math.cos(THETA)


# =============================================================================
# 2. Layer recovery, regularization, and sediment closure
# =============================================================================

@njit
def smooth_sign(value: float) -> float:
    return value / math.sqrt(value * value + SIGN_EPS * SIGN_EPS)


@njit
def bounded_inverse(depth: float, delta: float) -> float:
    """Return [1-exp(-depth/delta)]/depth with its finite zero limit."""
    if depth <= 1.0e-8 * delta:
        return 1.0 / delta
    return (1.0 - math.exp(-depth / delta)) / depth


@njit
def positive_part(value: float) -> float:
    """Smooth approximation of max(value, 0)."""
    return 0.5 * (value + math.sqrt(value * value + THETA_SMOOTH**2))


@njit
def closure_layers(U: np.ndarray):
    """Recover h0 and hp, clipping only inside the constitutive closures."""
    h = max(U[0], 2.0 * LAYER_FLOOR, DEPTH_FLOOR)
    q = U[1]
    up = U[2]

    if abs(up) < VELOCITY_FLOOR:
        up = VELOCITY_FLOOR if up >= 0.0 else -VELOCITY_FLOOR

    h0 = 3.0 * (h - q / up)
    h0 = min(max(h0, LAYER_FLOOR), h - LAYER_FLOOR)
    hp = h - h0
    return h, q, up, h0, hp


@njit
def basal_stress_hat(U: np.ndarray) -> float:
    """tau_b/[rho_m g H sin(theta)] = alpha sign(u_p)+2u_p/h0."""
    _, _, up, h0, _ = closure_layers(U)
    return ALPHA * smooth_sign(up) + 2.0 * up * bounded_inverse(h0, DELTA_H0)


@njit
def shields_parameter(U: np.ndarray) -> float:
    denominator = (DENSITY_RATIO - 1.0) * D_REL
    return abs(basal_stress_hat(U)) * SIN_THETA / denominator


@njit
def bedload_hat(U: np.ndarray) -> float:
    """Dimensionless Rickenmann bed-load discharge.

    q_b_hat = 3.1 (beta cos(theta))^0.05 (d_m/H)^1.5
              (d90/d30)^0.2 sqrt(Theta_b) [Theta_b-Theta_c]_+
              |q/h^(3/2)|^1.1 sign(u_p).
    """
    if not MOBILE_BED:
        return 0.0

    h, q, up, _, _ = closure_layers(U)
    theta_b = shields_parameter(U)
    excess = positive_part(theta_b - THETA_CRIT)

    coefficient = (
        3.1
        * (BETA * COS_THETA) ** 0.05
        * D_REL**1.5
        * D90_D30**0.2
    )
    local_froude_factor = abs(q) / h**1.5

    return (
        smooth_sign(up)
        * coefficient
        * math.sqrt(theta_b)
        * excess
        * local_froude_factor**1.1
    )


# =============================================================================
# 3. PyPDE callbacks: F, B, and S
# =============================================================================

def flux(U: np.ndarray) -> np.ndarray:
    h, q, up, b = U

    out = np.empty(4)
    out[0] = q
    out[1] = (
        7.0 * q * up / 5.0
        - 2.0 * h * up * up / 5.0
        + h * h / (2.0 * BETA)
    )
    out[2] = 0.5 * up * up + (h + b) / BETA
    out[3] = bedload_hat(U) / (1.0 - BED_POROSITY)
    return out


def nonconservative(U: np.ndarray) -> np.ndarray:
    """The sole nonconservative product is (h/beta) b_x in momentum."""
    out = np.zeros((4, 4))
    out[1, 3] = U[0] / BETA
    return out


def source(U: np.ndarray) -> np.ndarray:
    h, _, up, h0, hp = closure_layers(U)
    direction = smooth_sign(up)

    out = np.zeros(4)
    out[1] = (
        h
        - ALPHA * direction
        - 2.0 * up * bounded_inverse(h0, DELTA_H0)
    ) / BETA
    out[2] = (
        1.0
        - ALPHA * direction * bounded_inverse(hp, DELTA_HP)
    ) / BETA
    return out


# =============================================================================
# 4. Initial condition and solver
# =============================================================================

def uniform_state() -> np.ndarray:
    """Exact uniform Liu-Mei state under the Liu-Mei scaling."""
    h = 1.0
    h0 = 1.0 - ALPHA
    up = 0.5 * (1.0 - ALPHA) ** 2
    q = up * (h - h0 / 3.0)
    return np.array([h, q, up, 0.0], dtype=float)


def make_initial_condition():
    """One periodic wavelength with a small, consistent depth perturbation."""
    length = 2.0 * math.pi / WAVENUMBER
    dx = length / NX
    x = (np.arange(NX, dtype=float) + 0.5) * dx

    base = uniform_state()
    h0_base = 1.0 - ALPHA
    up_base = base[2]
    wave = np.cos(WAVENUMBER * x)

    Q0 = np.empty((NX, 4), dtype=float)
    Q0[:, 0] = 1.0 + PERTURBATION_AMPLITUDE * wave
    Q0[:, 1] = up_base * (Q0[:, 0] - h0_base / 3.0)
    Q0[:, 2] = up_base
    Q0[:, 3] = INITIAL_BED_AMPLITUDE * wave
    return x, Q0, length


def validate_parameters() -> None:
    """Catch inconsistent user inputs before the native solver starts."""
    if not 0.0 < ALPHA < 1.0:
        raise ValueError("ALPHA must lie in (0, 1).")
    if BETA <= 0.0 or BED_SLOPE <= 0.0:
        raise ValueError("BETA and BED_SLOPE must be positive.")
    if not 0.0 <= BED_POROSITY < 1.0:
        raise ValueError("BED_POROSITY must lie in [0, 1).")
    if DENSITY_RATIO <= 1.0:
        raise ValueError("DENSITY_RATIO must exceed one.")
    if D_REL <= 0.0 or D90_D30 <= 0.0:
        raise ValueError("D_REL and D90_D30 must be positive.")


def run_simulation():
    validate_parameters()

    try:
        from pypde import pde_solver
    except ImportError as exc:
        raise RuntimeError("PyPDE is not installed in this environment.") from exc

    x, Q0, length = make_initial_condition()

    print("Running dimensionless Liu-Mei-Exner model")
    print("  mobile bed   :", MOBILE_BED)
    print("  cells        :", NX)
    print("  domain       :", length)
    print("  final time   :", FINAL_TIME)

    out = pde_solver(
        Q0,
        FINAL_TIME,
        [length],
        F=flux,
        B=nonconservative,
        S=source,
        boundaryTypes="periodic",
        cfl=CFL,
        order=RECONSTRUCTION_ORDER,
        ndt=N_OUTPUTS,
        flux=NUMERICAL_FLUX,
        stiff=STIFF_SOURCE,
        nThreads=N_THREADS,
    )

    times = FINAL_TIME * np.arange(1, out.shape[0] + 1) / out.shape[0]
    return x, times, Q0, out


def parameter_record() -> dict:
    """Values written alongside the numerical solution."""
    base = uniform_state()
    return {
        "alpha": ALPHA,
        "beta": BETA,
        "bed_slope": BED_SLOPE,
        "theta_radians": THETA,
        "bed_porosity": BED_POROSITY,
        "density_ratio": DENSITY_RATIO,
        "d_m_over_H": D_REL,
        "d90_over_d30": D90_D30,
        "theta_crit": THETA_CRIT,
        "mobile_bed": MOBILE_BED,
        "uniform_h": base[0],
        "uniform_q": base[1],
        "uniform_u_p": base[2],
        "uniform_froude": math.sqrt(BETA) * base[1],
        "nx": NX,
        "final_time": FINAL_TIME,
        "n_outputs": N_OUTPUTS,
        "cfl": CFL,
        "order": RECONSTRUCTION_ORDER,
        "numerical_flux": NUMERICAL_FLUX,
        "stiff_source": STIFF_SOURCE,
        "n_threads": N_THREADS,
    }


def main() -> None:
    x, times, Q0, out = run_simulation()

    from bingham_roll_wave_postprocess import postprocess_solution

    postprocess_solution(
        x=x,
        times=times,
        Q0=Q0,
        out=out,
        output_directory=OUTPUT_DIRECTORY,
        snapshot_times=SNAPSHOT_TIMES,
        bed_porosity=BED_POROSITY,
        velocity_floor=VELOCITY_FLOOR,
        depth_floor=DEPTH_FLOOR,
        basal_stress_function=basal_stress_hat,
        shields_function=shields_parameter,
        bedload_function=bedload_hat,
        parameters=parameter_record(),
        save_text=SAVE_TEXT,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        dpi=PLOT_DPI,
    )


if __name__ == "__main__":
    main()
