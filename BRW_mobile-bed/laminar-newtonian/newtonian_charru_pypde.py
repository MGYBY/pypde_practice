#!/usr/bin/env python3
"""Laminar Newtonian roll waves over an erodible bed with Charru transport.

Dimensionless state
-------------------
    Q = [h, q, b, m]

where
    h : fluid depth,
    q : depth-integrated fluid discharge,
    b : packed-bed elevation relative to the mean inclined plane,
    m : mobile solid volume per unit bed area divided by the reference depth.

The equations are supplied to PyPDE as

    Q_t + F(Q)_x + B(Q) Q_x = S(Q).

The hydrodynamics are the Newtonian (n=1) limit of the Ng-Mei power-law
model and of the Liu-Mei Bingham model after the vanishing plug layer has
been eliminated.  The sediment equations follow the erosion-transport-
deposition model of Charru, Mouilleron & Eiff (2004) and Charru & Hinch
(2006).

All physical inputs in Section 1 are dimensionless.  Plotting and file output
are kept in the separate module ``newtonian_charru_postprocess.py``.
"""

from pathlib import Path
import math

import numpy as np
from numba import njit


# =============================================================================
# 1. Dimensionless physical and closure parameters (edit these first)
# =============================================================================

# Hydrodynamic case parameters.
# FROUDE > 1/sqrt(3) is unstable in the two-equation Ng-Mei/Shkadov model.
FROUDE = 0.7
BED_SLOPE = 0.05                 # S0 = tan(theta)

# Sediment and packed-bed parameters.
DENSITY_RATIO = 2.65               # s = rho_s/rho
RELATIVE_GRAIN_SIZE = 0.030        # delta = d/H
BED_POROSITY = 0.40                # lambda_p; packed solid fraction = 1-lambda_p

# Charru coefficients for an armoured/saturated bed.
# SHIELDS_THRESHOLD = 0.12           # theta_t0
SHIELDS_THRESHOLD = 0.040           # theta_t0
CHARRU_CE = 0.055                  # erosion coefficient
CHARRU_CD = 0.067                  # tau_p = d/(c_d V_s)
CHARRU_CU = 3.3                    # mobile-particle speed coefficient

# By default Lambda=T/tau_p is derived from the above dimensionless inputs and
# the Nusselt compatibility relation Re=3 Fr^2/S0.  Set this switch to False
# only when an independently calibrated adaptation time is available.
USE_DERIVED_RELAXATION_RATE = True
RELAXATION_RATE_OVERRIDE = 1.0     # used only when the switch above is False

# Enable/disable exchange and bed evolution without changing the state layout.
MOBILE_BED = True


# =============================================================================
# 2. Initial-value problem and PyPDE controls
# =============================================================================

WAVENUMBER = 1.20
DEPTH_PERTURBATION = 5.0e-3
INITIAL_BED_AMPLITUDE = 0.0

NX = 100
FINAL_TIME = 10.0
N_OUTPUTS = 160
CFL = 0.450
# RECONSTRUCTION_ORDER = 1
RECONSTRUCTION_ORDER = 2
NUMERICAL_FLUX = "rusanov"
# STIFF_SOURCE = True
STIFF_SOURCE = False
N_THREADS = 1

# Small closure safeguards.  The model is not intended for wetting/drying.
DEPTH_FLOOR = 1.0e-8
FLOW_DIRECTION_EPS = 1.0e-10
THRESHOLD_SMOOTH = 1.0e-8

# Output controls.
OUTPUT_DIRECTORY = Path(".")
SNAPSHOT_TIMES = np.linspace(0.0, FINAL_TIME, 21, dtype=float)
SAVE_TEXT = True
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_DPI = 220


# =============================================================================
# 3. Derived dimensionless groups
# =============================================================================

INCLINATION = math.atan(BED_SLOPE)
SIN_INCLINATION = math.sin(INCLINATION)
COS_INCLINATION = math.cos(INCLINATION)
PACKING_FRACTION = 1.0 - BED_POROSITY

FROUDE_SQUARED = FROUDE * FROUDE
REYNOLDS = 3.0 * FROUDE_SQUARED / BED_SLOPE
LENGTH_TO_DEPTH = FROUDE_SQUARED / BED_SLOPE   # L/H
DEPTH_TO_LENGTH = 1.0 / LENGTH_TO_DEPTH         # H/L

# Charru's original Stokes settling speed uses full g.  For a steep incline,
# replacing it by the bed-normal component would multiply Lambda by cos(theta).
DERIVED_RELAXATION_RATE = (
    CHARRU_CD
    * (DENSITY_RATIO - 1.0)
    * REYNOLDS
    * RELATIVE_GRAIN_SIZE
    / (18.0 * SIN_INCLINATION)
)
RELAXATION_RATE = (
    DERIVED_RELAXATION_RATE
    if USE_DERIVED_RELAXATION_RATE
    else RELAXATION_RATE_OVERRIDE
)

PARTICLE_VOLUME_FACTOR = math.pi / 6.0
UNIFORM_PARTICLE_REYNOLDS = 3.0 * REYNOLDS * RELATIVE_GRAIN_SIZE**2
GALILEO_NUMBER = (
    (DENSITY_RATIO - 1.0)
    * RELATIVE_GRAIN_SIZE**3
    * REYNOLDS**2
    / (FROUDE_SQUARED * COS_INCLINATION)
)


# =============================================================================
# 4. Local closures (Numba-compatible helpers)
# =============================================================================

@njit
def smooth_sign(value: float) -> float:
    """Smooth flow direction, equal to sign(value) away from zero."""
    return value / math.sqrt(value * value + FLOW_DIRECTION_EPS**2)


@njit
def positive_part(value: float) -> float:
    """Smooth approximation of max(value, 0)."""
    root = math.sqrt(value * value + THRESHOLD_SMOOTH**2)
    if value >= 0.0:
        return 0.5 * (value + root)
    # Equivalent algebraic branch avoids cancellation for large negative value.
    return 0.5 * THRESHOLD_SMOOTH**2 / (root - value)


@njit
def safe_depth(value: float) -> float:
    """Positive depth used only to prevent invalid closure arithmetic."""
    return max(value, DEPTH_FLOOR)


@njit
def basal_stress_ratio(Q: np.ndarray) -> float:
    """tau_b/(rho g H sin(theta)) = q/h^2 in the chosen Nusselt scaling."""
    h = safe_depth(Q[0])
    return Q[1] / (h * h)


@njit
def shields_number(Q: np.ndarray) -> float:
    """Local grain Shields number based on the Newtonian basal traction."""
    coefficient = SIN_INCLINATION / (
        (DENSITY_RATIO - 1.0) * RELATIVE_GRAIN_SIZE
    )
    return coefficient * abs(basal_stress_ratio(Q))


@njit
def normalized_shields(Q: np.ndarray) -> float:
    return shields_number(Q) / SHIELDS_THRESHOLD


@njit
def equilibrium_mobile_storage(Q: np.ndarray) -> float:
    """Dimensionless local-equilibrium mobile solid storage m_eq."""
    if not MOBILE_BED:
        return 0.0
    excess = positive_part(normalized_shields(Q) - 1.0)
    return (
        PARTICLE_VOLUME_FACTOR
        * CHARRU_CE
        * RELATIVE_GRAIN_SIZE
        * excess
    )


@njit
def charru_terms(Q: np.ndarray):
    """Return u_s, E, D, theta_b, Theta, and m_eq.

    All quantities are dimensionless in the Nusselt/Ng-Mei scaling:
      u_s : mobile-particle speed divided by U,
      E,D : solid-volume exchange rates multiplied by T/H,
      m_eq: mobile solid volume per bed area divided by H.
    """
    if not MOBILE_BED:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    h = safe_depth(Q[0])
    q = Q[1]
    mobile = max(Q[3], 0.0)

    stress_ratio = q / (h * h)
    theta_b = (
        SIN_INCLINATION
        * abs(stress_ratio)
        / ((DENSITY_RATIO - 1.0) * RELATIVE_GRAIN_SIZE)
    )
    theta_ratio = theta_b / SHIELDS_THRESHOLD
    excess = positive_part(theta_ratio - 1.0)

    direction = smooth_sign(q)
    particle_speed = (
        direction
        * CHARRU_CU
        * RELAXATION_RATE
        * RELATIVE_GRAIN_SIZE
        * DEPTH_TO_LENGTH
        * theta_ratio
    )

    mobile_equilibrium = (
        PARTICLE_VOLUME_FACTOR
        * CHARRU_CE
        * RELATIVE_GRAIN_SIZE
        * excess
    )
    erosion = RELAXATION_RATE * mobile_equilibrium
    deposition = RELAXATION_RATE * mobile

    return (
        particle_speed,
        erosion,
        deposition,
        theta_b,
        theta_ratio,
        mobile_equilibrium,
    )


# =============================================================================
# 5. PyPDE callbacks for Q=[h,q,b,m]
# =============================================================================

def flux(Q: np.ndarray) -> np.ndarray:
    """Conservative flux vector F(Q)."""
    h = max(Q[0], DEPTH_FLOOR)
    q = Q[1]
    mobile = max(Q[3], 0.0)
    particle_speed, _, _, _, _, _ = charru_terms(Q)

    out = np.empty(4)
    out[0] = q
    out[1] = 6.0 * q * q / (5.0 * h) + h * h / (2.0 * FROUDE_SQUARED)
    out[2] = 0.0
    out[3] = particle_speed * mobile
    return out


def nonconservative(Q: np.ndarray) -> np.ndarray:
    """Moving-bed pressure product (h/Fr^2) b_x in the momentum row."""
    out = np.zeros((4, 4))
    out[1, 2] = Q[0] / FROUDE_SQUARED
    return out


def source(Q: np.ndarray) -> np.ndarray:
    """Gravity/friction balance and packed/mobile sediment exchange."""
    h = safe_depth(Q[0])
    q = Q[1]
    _, erosion, deposition, _, _, _ = charru_terms(Q)

    out = np.zeros(4)
    out[1] = h - q / (h * h)
    out[2] = (deposition - erosion) / PACKING_FRACTION
    out[3] = erosion - deposition
    return out


# =============================================================================
# 6. Initial condition, validation, solver call, and output
# =============================================================================

def uniform_state() -> np.ndarray:
    """Steady Nusselt state, with mobile storage initially at local equilibrium."""
    state = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    state[3] = equilibrium_mobile_storage(state)
    return state


def make_initial_condition():
    """One periodic wavelength with fixed mean depth and discharge."""
    length = 2.0 * math.pi / WAVENUMBER
    dx = length / NX
    x = (np.arange(NX, dtype=float) + 0.5) * dx
    wave = np.cos(WAVENUMBER * x)

    Q0 = np.empty((NX, 4), dtype=float)
    Q0[:, 0] = 1.0 + DEPTH_PERTURBATION * wave
    Q0[:, 1] = 1.0
    Q0[:, 2] = INITIAL_BED_AMPLITUDE * wave

    # Start the mobile inventory at its local equilibrium value so the initial
    # condition does not contain an artificial sediment-exchange impulse.
    for i in range(NX):
        trial = np.array([Q0[i, 0], Q0[i, 1], Q0[i, 2], 0.0])
        Q0[i, 3] = equilibrium_mobile_storage(trial)

    return x, Q0, length


def validate_parameters() -> None:
    if FROUDE <= 0.0 or BED_SLOPE <= 0.0:
        raise ValueError("FROUDE and BED_SLOPE must be positive.")
    if DENSITY_RATIO <= 1.0:
        raise ValueError("DENSITY_RATIO must exceed one.")
    if RELATIVE_GRAIN_SIZE <= 0.0:
        raise ValueError("RELATIVE_GRAIN_SIZE must be positive.")
    if not 0.0 <= BED_POROSITY < 1.0:
        raise ValueError("BED_POROSITY must lie in [0,1).")
    if SHIELDS_THRESHOLD <= 0.0:
        raise ValueError("SHIELDS_THRESHOLD must be positive.")
    if min(CHARRU_CE, CHARRU_CD, CHARRU_CU) <= 0.0:
        raise ValueError("The Charru coefficients must be positive.")
    if MOBILE_BED and RELAXATION_RATE <= 0.0:
        raise ValueError("RELAXATION_RATE must be positive for a mobile bed.")
    if NX < 4 or FINAL_TIME <= 0.0 or N_OUTPUTS < 1:
        raise ValueError("Invalid grid or time-output settings.")


def parameter_record() -> dict:
    base = uniform_state()
    terms = charru_terms(base)
    return {
        "state_order": "h q b m",
        "froude": FROUDE,
        "bed_slope_tan_theta": BED_SLOPE,
        "inclination_radians": INCLINATION,
        "density_ratio": DENSITY_RATIO,
        "relative_grain_size_d_over_H": RELATIVE_GRAIN_SIZE,
        "bed_porosity": BED_POROSITY,
        "packing_fraction": PACKING_FRACTION,
        "shields_threshold": SHIELDS_THRESHOLD,
        "charru_ce": CHARRU_CE,
        "charru_cd": CHARRU_CD,
        "charru_cu": CHARRU_CU,
        "reynolds_derived": REYNOLDS,
        "L_over_H": LENGTH_TO_DEPTH,
        "relaxation_rate_T_over_tau": RELAXATION_RATE,
        "relaxation_rate_is_derived": USE_DERIVED_RELAXATION_RATE,
        "uniform_particle_reynolds": UNIFORM_PARTICLE_REYNOLDS,
        "galileo_number": GALILEO_NUMBER,
        "mobile_bed": MOBILE_BED,
        "uniform_h": base[0],
        "uniform_q": base[1],
        "uniform_b": base[2],
        "uniform_m": base[3],
        "uniform_shields": terms[3],
        "uniform_normalized_shields": terms[4],
        "uniform_particle_speed": terms[0],
        "wavenumber": WAVENUMBER,
        "depth_perturbation": DEPTH_PERTURBATION,
        "initial_bed_amplitude": INITIAL_BED_AMPLITUDE,
        "nx": NX,
        "final_time": FINAL_TIME,
        "n_outputs": N_OUTPUTS,
        "cfl": CFL,
        "reconstruction_order": RECONSTRUCTION_ORDER,
        "numerical_flux": NUMERICAL_FLUX,
        "stiff_source": STIFF_SOURCE,
        "n_threads": N_THREADS,
        "depth_floor": DEPTH_FLOOR,
        "direction_epsilon": FLOW_DIRECTION_EPS,
        "threshold_smoothing": THRESHOLD_SMOOTH,
    }


def print_case_summary() -> None:
    base = uniform_state()
    particle_speed, erosion, deposition, theta_b, theta_ratio, m_eq = charru_terms(base)
    critical_froude = 1.0 / math.sqrt(3.0)

    print("Dimensionless Newtonian-Charru shallow-layer model")
    print("  state                         : [h, q, b, m]")
    print("  Froude number                :", FROUDE)
    print("  Ng-Mei critical Froude       :", critical_froude)
    print("  derived Reynolds number      :", REYNOLDS)
    print("  uniform particle Reynolds    :", UNIFORM_PARTICLE_REYNOLDS)
    print("  Galileo number               :", GALILEO_NUMBER)
    print("  relaxation rate T/tau_p      :", RELAXATION_RATE)
    print("  uniform Shields              :", theta_b)
    print("  uniform Theta=Shields/theta_t:", theta_ratio)
    print("  uniform m_eq                 :", m_eq)
    print("  uniform particle speed       :", particle_speed)
    print("  uniform E, D                 :", erosion, deposition)

    if FROUDE <= critical_froude:
        print("  NOTE: the uniform two-equation flow is linearly stable in the")
        print("        Ng-Mei first-order shallow-layer approximation.")
    if theta_b > 0.24 or UNIFORM_PARTICLE_REYNOLDS > 0.30:
        print("  WARNING: the uniform sediment state is outside or at the edge of")
        print("           the Charru et al. (2004) experimental range.")


def run_simulation():
    validate_parameters()
    print_case_summary()

    try:
        from pypde import pde_solver
    except ImportError as exc:
        raise RuntimeError(
            "PyPDE is not installed. Create the supplied Python 3.8 environment "
            "and install/build PyPDE before running this file."
        ) from exc

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
        order=RECONSTRUCTION_ORDER,
        ndt=N_OUTPUTS,
        flux=NUMERICAL_FLUX,
        stiff=STIFF_SOURCE,
        nThreads=N_THREADS,
    )
    times = FINAL_TIME * np.arange(1, out.shape[0] + 1, dtype=float) / out.shape[0]
    return x, times, Q0, out


def main() -> None:
    x, times, Q0, out = run_simulation()

    from newtonian_charru_postprocess import postprocess_solution

    postprocess_solution(
        x=x,
        times=times,
        Q0=Q0,
        out=out,
        output_directory=OUTPUT_DIRECTORY,
        snapshot_times=SNAPSHOT_TIMES,
        bed_porosity=BED_POROSITY,
        depth_floor=DEPTH_FLOOR,
        basal_stress_function=basal_stress_ratio,
        charru_terms_function=charru_terms,
        parameters=parameter_record(),
        save_text=SAVE_TEXT,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        dpi=PLOT_DPI,
    )


if __name__ == "__main__":
    main()
