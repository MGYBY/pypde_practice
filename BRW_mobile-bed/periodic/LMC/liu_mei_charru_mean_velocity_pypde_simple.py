#!/usr/bin/env python3
"""Liu–Mei–Charru model solved with PyPDE.

Dimensionless state
-------------------
    U = [h, q, u_p, b, m]

The velocity scale is the steady-uniform depth-averaged velocity.  The
Charru flight time is evaluated locally from a Stokes-type settling speed
using the apparent Bingham viscosity at the bed.

Edit only the parameter block below for normal runs.  Plotting and file
output are handled by ``liu_mei_charru_mean_velocity_postprocess_simple.py``.
"""

from pathlib import Path
import math

import numpy as np
from numba import njit


# =============================================================================
# 1. User inputs
# =============================================================================

# Physical/closure parameters
ALPHA = 0.30                      # tau_y/[rho g H sin(theta)]
FROUDE = 0.953                    # U_bar/sqrt[g H cos(theta)]
BED_SLOPE = 0.02                 # tan(theta)
MOBILE_BED = True
# MOBILE_BED = False
BED_POROSITY = 0.40
DENSITY_RATIO = 2650.0 / 1200.0  # rho_s/rho_m
GRAIN_TO_DEPTH = 0.10            # d/H
SHIELDS_THRESHOLD = 0.12
CHARRU_CE = 0.055
CHARRU_CD = 0.067
CHARRU_CU = 3.3

# Initial disturbance
WAVENUMBER = 1.20
DEPTH_PERTURBATION = 1.0e-2
BED_PERTURBATION = 0.0

# PyPDE settings
NX = 128
FINAL_TIME = 40.0
N_OUTPUTS = 21
CFL = 0.40
ORDER = 2
NUMERICAL_FLUX = "rusanov"
# NUMERICAL_FLUX = "hll"
# STIFF_SOURCE = True
STIFF_SOURCE = False
N_THREADS = 1

# Closure-only regularization
DEPTH_FLOOR = 1.0e-8
VELOCITY_FLOOR = 1.0e-10
LAYER_FLOOR = 1.0e-8
DELTA_H0 = 1.0e-2
DELTA_HP = 5.0e-3
SIGN_EPS = 1.0e-10
SHIELDS_SMOOTH = 1.0e-8

# Output settings
# N_OUTPUTS (= PyPDE ndt) controls how many solution snapshots are returned.
# Every returned snapshot is written by the postprocessor.
OUTPUT_DIRECTORY = Path("liu_mei_charru_mean_velocity_output")
SAVE_SNAPSHOT_TEXT = True
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_DPI = 220


# =============================================================================
# 2. Derived constants
# =============================================================================

INCLINATION = math.atan(BED_SLOPE)
SIN_THETA = math.sin(INCLINATION)
PACKING_FRACTION = 1.0 - BED_POROSITY
INV_FR2 = 1.0 / FROUDE**2
GRAIN_TO_LENGTH = GRAIN_TO_DEPTH * BED_SLOPE  # d/L, L=H/tan(theta)
PARTICLE_VOLUME_FACTOR = math.pi / 6.0

# U_bar = C_ALPHA rho g H^2 sin(theta)/eta_B
C_ALPHA = (1.0 - ALPHA) ** 2 * (2.0 + ALPHA) / 6.0
R_BINGHAM = C_ALPHA * INV_FR2
SHIELDS_FACTOR = SIN_THETA / ((DENSITY_RATIO - 1.0) * GRAIN_TO_DEPTH)
SETTLING_FACTOR = (
    (DENSITY_RATIO - 1.0)
    * GRAIN_TO_DEPTH**2
    / (18.0 * C_ALPHA * SIN_THETA)
)
RELAXATION_FACTOR = CHARRU_CD / (GRAIN_TO_DEPTH * BED_SLOPE)
MOBILE_STORAGE_FACTOR = PARTICLE_VOLUME_FACTOR * CHARRU_CE * GRAIN_TO_DEPTH


# =============================================================================
# 3. Local closures
# =============================================================================

@njit
def smooth_sign(value):
    return value / math.sqrt(value * value + SIGN_EPS * SIGN_EPS)


@njit
def bounded_inverse(depth, delta):
    """Smooth approximation of 1/depth with limit 1/delta."""
    if depth <= 1.0e-8 * delta:
        return 1.0 / delta
    return -math.expm1(-depth / delta) / depth


@njit
def positive_part(value):
    """Smooth and cancellation-resistant approximation of max(value, 0)."""
    root = math.sqrt(value * value + SHIELDS_SMOOTH * SHIELDS_SMOOTH)
    if value >= 0.0:
        return 0.5 * (value + root)
    return 0.5 * SHIELDS_SMOOTH * SHIELDS_SMOOTH / (root - value)


@njit
def model_terms(U):
    """Return all regularized local variables in one evaluation.

    Returns
    -------
    h, q, up, h0, hp, mobile,
    tau_hat, shields, theta_ratio, eta_ratio, vs_ratio, relaxation,
    particle_speed, erosion, deposition
    """
    h = max(max(U[0], DEPTH_FLOOR), 2.0 * LAYER_FLOOR)
    q = U[1]
    up = U[2]
    if abs(up) < VELOCITY_FLOOR:
        up = VELOCITY_FLOOR if up >= 0.0 else -VELOCITY_FLOOR

    h0 = 3.0 * (h - q / up)
    h0 = min(max(h0, LAYER_FLOOR), h - LAYER_FLOOR)
    hp = h - h0
    mobile = max(U[4], 0.0)

    direction = smooth_sign(up)
    tau_hat = (
        ALPHA * direction
        + 2.0 * C_ALPHA * up * bounded_inverse(h0, DELTA_H0)
    )
    shields = abs(tau_hat) * SHIELDS_FACTOR
    theta_ratio = shields / SHIELDS_THRESHOLD

    eta_ratio = 1.0 + ALPHA * h0 / (2.0 * C_ALPHA * max(abs(up), VELOCITY_FLOOR))
    vs_ratio = SETTLING_FACTOR / eta_ratio
    relaxation = RELAXATION_FACTOR * vs_ratio

    if MOBILE_BED:
        excess = positive_part(theta_ratio - 1.0)
        particle_speed = (
            smooth_sign(tau_hat)
            * CHARRU_CU
            * relaxation
            * GRAIN_TO_LENGTH
            * theta_ratio
        )
        erosion = relaxation * MOBILE_STORAGE_FACTOR * excess
        deposition = relaxation * mobile
    else:
        particle_speed = 0.0
        erosion = 0.0
        deposition = 0.0

    return (
        h, q, up, h0, hp, mobile,
        tau_hat, shields, theta_ratio, eta_ratio, vs_ratio, relaxation,
        particle_speed, erosion, deposition,
    )


@njit
def diagnostic_values(U):
    """Nine closure quantities used by the postprocessor."""
    terms = model_terms(U)
    out = np.empty(9)
    out[0] = terms[6]   # tau_hat
    out[1] = terms[7]   # Shields
    out[2] = terms[8]   # Theta
    out[3] = terms[9]   # eta_bed/eta_B
    out[4] = terms[10]  # V_s/U_bar
    out[5] = terms[11]  # T/tau_p
    out[6] = terms[12]  # particle speed
    out[7] = terms[13]  # erosion
    out[8] = terms[14]  # deposition
    return out


@njit
def equilibrium_mobile_storage(U):
    if not MOBILE_BED:
        return 0.0
    theta_ratio = model_terms(U)[8]
    return MOBILE_STORAGE_FACTOR * positive_part(theta_ratio - 1.0)


# =============================================================================
# 4. PyPDE callbacks: U_t + F(U)_x + B(U)U_x = S(U)
# =============================================================================

def flux(U):
    # Hydrodynamic fluxes use the raw evolved state; regularization is confined
    # to the rheology and sediment closures, as in the original implementation.
    h, q, up, bed, mobile_raw = U
    particle_speed = model_terms(U)[12]

    F = np.empty(5)
    F[0] = q
    F[1] = 7.0 * q * up / 5.0 - 2.0 * h * up * up / 5.0 + 0.5 * h * h * INV_FR2
    F[2] = 0.5 * up * up + (h + bed) * INV_FR2
    F[3] = 0.0
    F[4] = particle_speed * max(mobile_raw, 0.0)
    return F


def nonconservative(U):
    B = np.zeros((5, 5))
    B[1, 3] = U[0] * INV_FR2
    return B


def source(U):
    h, _, up, h0, hp, _, _, _, _, _, _, _, _, erosion, deposition = model_terms(U)
    direction = smooth_sign(up)

    S = np.zeros(5)
    S[1] = (
        (h - ALPHA * direction) * INV_FR2
        - 2.0 * R_BINGHAM * up * bounded_inverse(h0, DELTA_H0)
    )
    S[2] = (1.0 - ALPHA * direction * bounded_inverse(hp, DELTA_HP)) * INV_FR2
    S[3] = (deposition - erosion) / PACKING_FRACTION
    S[4] = erosion - deposition
    return S


# =============================================================================
# 5. Initial condition and run
# =============================================================================

def uniform_state():
    U = np.array([1.0, 1.0, 3.0 / (2.0 + ALPHA), 0.0, 0.0])
    U[4] = equilibrium_mobile_storage(U)
    return U


def make_initial_condition():
    length = 2.0 * math.pi / WAVENUMBER
    x = (np.arange(NX) + 0.5) * length / NX
    wave = np.cos(WAVENUMBER * x)

    up0 = 3.0 / (2.0 + ALPHA)
    h0 = 1.0 - ALPHA

    Q0 = np.empty((NX, 5))
    Q0[:, 0] = 1.0 + DEPTH_PERTURBATION * wave
    Q0[:, 1] = up0 * (Q0[:, 0] - h0 / 3.0)
    Q0[:, 2] = up0
    Q0[:, 3] = BED_PERTURBATION * wave

    for i in range(NX):
        trial = np.array([Q0[i, 0], Q0[i, 1], Q0[i, 2], Q0[i, 3], 0.0])
        Q0[i, 4] = equilibrium_mobile_storage(trial)

    return x, Q0, length


def validate_inputs():
    if not 0.0 < ALPHA < 1.0:
        raise ValueError("ALPHA must be in (0, 1).")
    if FROUDE <= 0.0 or BED_SLOPE <= 0.0:
        raise ValueError("FROUDE and BED_SLOPE must be positive.")
    if not 0.0 <= BED_POROSITY < 1.0:
        raise ValueError("BED_POROSITY must be in [0, 1).")
    if DENSITY_RATIO <= 1.0 or GRAIN_TO_DEPTH <= 0.0:
        raise ValueError("DENSITY_RATIO must exceed 1 and GRAIN_TO_DEPTH must be positive.")
    if min(SHIELDS_THRESHOLD, CHARRU_CE, CHARRU_CD, CHARRU_CU) <= 0.0:
        raise ValueError("Charru closure parameters must be positive.")
    if NX < 4 or FINAL_TIME <= 0.0 or N_OUTPUTS < 1:
        raise ValueError("Invalid numerical grid or output settings.")


def print_case_summary():
    U = uniform_state()
    d = diagnostic_values(U)
    print("Liu-Mei-Charru model, U=[h,q,u_p,b,m]")
    print(f"  mobile bed       : {MOBILE_BED}")
    print(f"  uniform state    : {U}")
    print(f"  C_alpha          : {C_ALPHA:.8g}")
    print(f"  R_Bingham        : {R_BINGHAM:.8g}")
    print(f"  Shields / Theta  : {d[1]:.8g} / {d[2]:.8g}")
    print(f"  eta_bed/eta_B    : {d[3]:.8g}")
    print(f"  V_s/U_bar        : {d[4]:.8g}")
    print(f"  T/tau_p          : {d[5]:.8g}")
    print(f"  particle speed   : {d[6]:.8g}")
    if MOBILE_BED and not 0.04 <= d[1] <= 0.24:
        print("  WARNING: uniform Shields is outside Charru et al.'s approximate 0.04-0.24 range.")


def run_simulation():
    validate_inputs()
    print_case_summary()

    try:
        from pypde import pde_solver
    except ImportError as exc:
        raise RuntimeError("PyPDE is not installed in this environment.") from exc

    x, Q0, length = make_initial_condition()
    out = pde_solver(
        Q0,
        FINAL_TIME,
        [length],
        F=flux,
        B=nonconservative,
        S=source,
        boundaryTypes="periodic",
        cfl=CFL,
        order=ORDER,
        ndt=N_OUTPUTS,
        flux=NUMERICAL_FLUX,
        stiff=STIFF_SOURCE,
        nThreads=N_THREADS,
    )
    return x, Q0, out


def parameter_record():
    U = uniform_state()
    d = diagnostic_values(U)
    return {
        "alpha": ALPHA,
        "froude": FROUDE,
        "bed_slope": BED_SLOPE,
        "bed_porosity": BED_POROSITY,
        "density_ratio": DENSITY_RATIO,
        "grain_to_depth": GRAIN_TO_DEPTH,
        "shields_threshold": SHIELDS_THRESHOLD,
        "charru_ce": CHARRU_CE,
        "charru_cd": CHARRU_CD,
        "charru_cu": CHARRU_CU,
        "mobile_bed": MOBILE_BED,
        "C_alpha": C_ALPHA,
        "R_Bingham": R_BINGHAM,
        "uniform_state": U.tolist(),
        "uniform_tau_hat": d[0],
        "uniform_shields": d[1],
        "uniform_eta_ratio": d[3],
        "uniform_settling_ratio": d[4],
        "uniform_relaxation": d[5],
        "nx": NX,
        "final_time": FINAL_TIME,
        "n_outputs": N_OUTPUTS,
        "cfl": CFL,
        "order": ORDER,
        "numerical_flux": NUMERICAL_FLUX,
        "stiff_source": STIFF_SOURCE,
        "n_threads": N_THREADS,
        "delta_h0": DELTA_H0,
        "delta_hp": DELTA_HP,
        "shields_smooth": SHIELDS_SMOOTH,
    }


def main():
    x, Q0, out = run_simulation()

    from liu_mei_charru_mean_velocity_postprocess_simple import postprocess_solution

    postprocess_solution(
        x=x,
        Q0=Q0,
        out=out,
        final_time=FINAL_TIME,
        n_outputs=N_OUTPUTS,
        output_directory=OUTPUT_DIRECTORY,
        bed_porosity=BED_POROSITY,
        velocity_floor=VELOCITY_FLOOR,
        depth_floor=DEPTH_FLOOR,
        diagnostic_function=diagnostic_values,
        parameters=parameter_record(),
        save_snapshot_text=SAVE_SNAPSHOT_TEXT,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        dpi=PLOT_DPI,
    )


if __name__ == "__main__":
    main()
