#!/usr/bin/env python3
"""Audited two-layer power-law Kármán–Pohlhausen PyPDE model.

The dimensionless conserved state is

    Q = [h_l, h_u, q_l, q_u]

and the governing equations are supplied to PyPDE as

    Q_t + F(Q)_x + B(Q) Q_x = S(Q).

Compared with the earlier implementation, this edition:

* preserves the true initial condition from PyPDE's in-place work buffer;
* uses a user-controlled output interval and explicit output-time array;
* evaluates the lower profile moments robustly as lambda -> 1;
* rejects, rather than clips or silently repairs, inadmissible states;
* rejects a failed traction-continuity root instead of using an endpoint;
* passes all constitutive parameters into Numba kernels at run time;
* supports a selected linear eigenmode as the default smooth disturbance;
* provides closure-residual and hyperbolicity diagnostics.

Physical branch
---------------
The implemented closure is deliberately restricted to

    h_l > 0, h_u > 0,
    tau_b > tau_I >= 0,
    U_I > 0,
    W_u = U_s - U_I >= 0.

Drying, reverse flow, and interfacial shear reversal require an extended
constitutive branch and are not silently approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from numba import njit


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# --- Dimensionless physical parameters --------------------------------------
FR_LOWER = 0.80
DEPTH_RATIO = 1.00
DENSITY_RATIO = 0.80
N_LOWER = 0.40
N_UPPER = 0.80
SCALED_CONSISTENCY_RATIO = 1.00

# --- Computational domain ---------------------------------------------------
DOMAIN_LENGTH = 5.0
NX = 150  # one wavelength and 150 cells per wavelength with the defaults
FINAL_TIME = 16.0

# Positive-time outputs are requested at OUTPUT_INTERVAL, 2*OUTPUT_INTERVAL,
# ..., FINAL_TIME.  The true initial state is saved separately at t=0.
OUTPUT_INTERVAL = 1.0

# "uniform_nominal": one efficient PyPDE call.  Intermediate frame labels are
# nominal targets and can overshoot by less than one internal CFL step.
# "exact_segmented": one ndt=1 PyPDE call per interval.  Each requested time is
# an exact segment endpoint, at the cost of repeated solver startup.
OUTPUT_MODE = "uniform_nominal"

# --- Initial disturbance ----------------------------------------------------
# Recommended for linear-stability comparisons:
#   "linear_eigenmode"
# Legacy reproduction options:
#   "legacy_local_normal_flow"
#   "legacy_froude_preserving"
# A simple state-space perturbation is also available:
#   "depth_only"
INITIAL_CONDITION_MODE = "legacy_froude_preserving"

# For a linear eigenmode, this is the peak interface amplitude for the
# interfacial branch or the peak free-surface amplitude for the free-surface
# branch.  Values around 1e-4--1e-3 are appropriate for linear tests.
PERTURBATION_AMPLITUDE = 0.05
PERTURBATION_WAVELENGTH = DOMAIN_LENGTH
EIGENMODE_BRANCH = "free_surface"  # "free_surface" or "interface"

# "periodic" or a smooth periodic-distance Gaussian packet.
PERTURBATION_ENVELOPE = "periodic"
PACKET_CENTER = 0.50 * DOMAIN_LENGTH
PACKET_WIDTH = 0.12 * DOMAIN_LENGTH

# These settings are used only by the legacy initial-condition modes.
UPPER_LAYER_PHASE_SHIFT = 0.0

# --- PyPDE settings ----------------------------------------------------------
CFL = 0.880
RECONSTRUCTION_ORDER = 2
NUMERICAL_FLUX = "rusanov"  # "rusanov", "roe", or "osher"
STIFF_SOURCE = False
N_THREADS = 15
BOUNDARY_TYPE = "periodic"

# --- Output -----------------------------------------------------------------
OUTPUT_DIRECTORY = Path("two_layer_powerlaw_output_v2")
OUTPUT_STEM = "two_layer_powerlaw_kp"
RUN_POSTPROCESSING = True

# --- Closure and diagnostic tolerances --------------------------------------
MIN_DEPTH = 1.0e-8
LAMBDA_MAX = 1.0 - 1.0e-12
LAMBDA_MAX_ITERATIONS = 60
CLOSURE_RESIDUAL_RTOL = 2.0e-13
CLOSURE_LAMBDA_XTOL = 2.0e-15
BRANCH_VELOCITY_TOL = 5.0e-13
HYPERBOLICITY_IMAG_TOL = 2.0e-7

# Initial-condition preflight.  Checking every cell is inexpensive for the
# default grid and catches invalid branches before PyPDE starts.
PREFLIGHT_EVERY_CELL = True


# =============================================================================
# 2. VALIDATION AND DERIVED MODEL CONSTANTS
# =============================================================================

_ALLOWED_FLUXES = {"rusanov", "roe", "osher"}
_ALLOWED_BOUNDARIES = {"periodic", "transitive"}
_ALLOWED_OUTPUT_MODES = {"uniform_nominal", "exact_segmented"}
_ALLOWED_IC_MODES = {
    "linear_eigenmode",
    "legacy_local_normal_flow",
    "legacy_froude_preserving",
    "depth_only",
}
_ALLOWED_BRANCHES = {"free_surface", "interface"}
_ALLOWED_ENVELOPES = {"periodic", "gaussian_packet"}


class ClosureError(RuntimeError):
    """Raised when a state leaves the declared positive-shear closure branch."""


@dataclass(frozen=True)
class DerivedModel:
    m_lower: float
    m_upper: float
    upper_a: float
    upper_b: float
    uniform_lambda: float
    uniform_c: float
    uniform_a: float
    uniform_b: float
    lambda_lower: float
    lambda_upper: float
    uniform_tau_b: float
    uniform_tau_i: float
    uniform_u_interface: float
    uniform_w_upper: float
    uniform_u_upper: float
    uniform_q_upper: float


def _validate_parameters() -> None:
    finite_positive = {
        "FR_LOWER": FR_LOWER,
        "DEPTH_RATIO": DEPTH_RATIO,
        "DENSITY_RATIO": DENSITY_RATIO,
        "SCALED_CONSISTENCY_RATIO": SCALED_CONSISTENCY_RATIO,
        "DOMAIN_LENGTH": DOMAIN_LENGTH,
        "FINAL_TIME": FINAL_TIME,
        "OUTPUT_INTERVAL": OUTPUT_INTERVAL,
        "PERTURBATION_WAVELENGTH": PERTURBATION_WAVELENGTH,
        "PACKET_WIDTH": PACKET_WIDTH,
    }
    invalid = [
        name
        for name, value in finite_positive.items()
        if (not np.isfinite(value)) or value <= 0.0
    ]
    if invalid:
        raise ValueError("These settings must be finite and positive: " + ", ".join(invalid))

    if not (0.0 < N_LOWER <= 1.0 and 0.0 < N_UPPER <= 1.0):
        raise ValueError("Power-law indices must satisfy 0 < n <= 1.")
    if NX < 8 or int(NX) != NX:
        raise ValueError("NX must be an integer of at least 8.")
    if not (0.0 < CFL < 1.0):
        raise ValueError("CFL must satisfy 0 < CFL < 1.")
    if int(RECONSTRUCTION_ORDER) != RECONSTRUCTION_ORDER or RECONSTRUCTION_ORDER < 1:
        raise ValueError("RECONSTRUCTION_ORDER must be a positive integer.")
    if NUMERICAL_FLUX.lower() not in _ALLOWED_FLUXES:
        raise ValueError(f"NUMERICAL_FLUX must be one of {sorted(_ALLOWED_FLUXES)}.")
    if BOUNDARY_TYPE not in _ALLOWED_BOUNDARIES:
        raise ValueError(
            f"BOUNDARY_TYPE must be one of {sorted(_ALLOWED_BOUNDARIES)} "
            "using PyPDE's exact lowercase spelling."
        )
    if OUTPUT_MODE not in _ALLOWED_OUTPUT_MODES:
        raise ValueError(f"OUTPUT_MODE must be one of {sorted(_ALLOWED_OUTPUT_MODES)}.")
    if INITIAL_CONDITION_MODE not in _ALLOWED_IC_MODES:
        raise ValueError(
            f"INITIAL_CONDITION_MODE must be one of {sorted(_ALLOWED_IC_MODES)}."
        )
    if EIGENMODE_BRANCH not in _ALLOWED_BRANCHES:
        raise ValueError(f"EIGENMODE_BRANCH must be one of {sorted(_ALLOWED_BRANCHES)}.")
    if PERTURBATION_ENVELOPE not in _ALLOWED_ENVELOPES:
        raise ValueError(
            f"PERTURBATION_ENVELOPE must be one of {sorted(_ALLOWED_ENVELOPES)}."
        )
    if not np.isfinite(PERTURBATION_AMPLITUDE) or PERTURBATION_AMPLITUDE < 0.0:
        raise ValueError("PERTURBATION_AMPLITUDE must be finite and non-negative.")

    n_intervals = FINAL_TIME / OUTPUT_INTERVAL
    if not np.isclose(n_intervals, round(n_intervals), rtol=0.0, atol=2.0e-12):
        raise ValueError("FINAL_TIME must be an integer multiple of OUTPUT_INTERVAL.")

    if BOUNDARY_TYPE == "periodic":
        wavelengths = DOMAIN_LENGTH / PERTURBATION_WAVELENGTH
        if not np.isclose(wavelengths, round(wavelengths), atol=2.0e-12, rtol=0.0):
            raise ValueError(
                "For periodic boundaries, DOMAIN_LENGTH/PERTURBATION_WAVELENGTH "
                "must be an integer."
            )


@njit(cache=True)
def _lower_shape_coefficients_core(lam: float, m: float) -> Tuple[float, float, float]:
    """Stable lower-profile coefficients C, A and B for 0 <= lambda < 1."""
    if lam <= 1.0e-14:
        c = 1.0 / m
        a = m / (m + 1.0)
        b = 1.0 - 2.0 / (m + 1.0) + 1.0 / (2.0 * m + 1.0)
        return c, a, b

    eps = 1.0 - lam
    if eps < 1.0e-3:
        # Fourth-order expansion around the constant-shear/Couette limit.
        c = (
            1.0
            - 0.5 * (m - 1.0) * eps
            + (m - 1.0) * (m - 2.0) * eps**2 / 6.0
            - (m - 1.0) * (m - 2.0) * (m - 3.0) * eps**3 / 24.0
            + (m - 1.0)
            * (m - 2.0)
            * (m - 3.0)
            * (m - 4.0)
            * eps**4
            / 120.0
        )
        a = (
            0.5
            + (m - 1.0) * eps / 12.0
            + (m - 1.0) * eps**2 / 24.0
            - (m - 1.0) * (m**2 - 19.0) * eps**3 / 720.0
            - (m - 3.0) * (m - 1.0) * (m + 3.0) * eps**4 / 480.0
        )
        b = (
            1.0 / 3.0
            + (m - 1.0) * eps / 12.0
            + (m - 1.0) * (m + 7.0) * eps**2 / 180.0
            - (m - 1.0) * (m**2 - 4.0 * m - 17.0) * eps**3 / 720.0
            - (m - 1.0)
            * (m + 2.0)
            * (m**2 + 8.0 * m - 41.0)
            * eps**4
            / 5040.0
        )
        return c, a, b

    log_lam = math.log(lam)
    one_minus_lam_m = -math.expm1(m * log_lam)
    j_m = -math.expm1((m + 1.0) * log_lam) / ((m + 1.0) * eps)
    j_2m = -math.expm1((2.0 * m + 1.0) * log_lam) / (
        (2.0 * m + 1.0) * eps
    )
    c = one_minus_lam_m / (m * eps)
    a = (1.0 - j_m) / one_minus_lam_m
    b = (1.0 - 2.0 * j_m + j_2m) / (one_minus_lam_m * one_minus_lam_m)
    return c, a, b


def lower_shape_coefficients(lam: float) -> Tuple[float, float, float]:
    """Public, checked lower-profile coefficient evaluator."""
    if not np.isfinite(lam) or not (0.0 <= lam < 1.0):
        raise ValueError("lambda must satisfy 0 <= lambda < 1.")
    c, a, b = _lower_shape_coefficients_core(float(lam), DERIVED.m_lower)
    values = np.asarray([c, a, b], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise FloatingPointError(
            f"Invalid lower shape coefficients at lambda={lam:.16g}: {values}."
        )
    return float(c), float(a), float(b)


def _derive_model() -> DerivedModel:
    m_l = (N_LOWER + 1.0) / N_LOWER
    m_u = (N_UPPER + 1.0) / N_UPPER
    a_u = m_u / (m_u + 1.0)
    b_u = 1.0 - 2.0 / (m_u + 1.0) + 1.0 / (2.0 * m_u + 1.0)

    uniform_lambda = (
        DENSITY_RATIO * DEPTH_RATIO
        / (1.0 + DENSITY_RATIO * DEPTH_RATIO)
    )
    c0, a0, b0 = _lower_shape_coefficients_core(uniform_lambda, m_l)

    lambda_l = (
        (1.0 + DENSITY_RATIO * DEPTH_RATIO)
        * (a0 * c0) ** N_LOWER
        / FR_LOWER**2
    )
    lambda_u = SCALED_CONSISTENCY_RATIO * lambda_l

    tau_b0 = (1.0 + DENSITY_RATIO * DEPTH_RATIO) / FR_LOWER**2
    tau_i0 = DENSITY_RATIO * DEPTH_RATIO / FR_LOWER**2
    ui0 = 1.0 / a0
    w0 = DEPTH_RATIO / m_u * (tau_i0 / lambda_u) ** (1.0 / N_UPPER)
    uu0 = ui0 + a_u * w0
    qu0 = DEPTH_RATIO * uu0

    return DerivedModel(
        m_lower=m_l,
        m_upper=m_u,
        upper_a=a_u,
        upper_b=b_u,
        uniform_lambda=uniform_lambda,
        uniform_c=c0,
        uniform_a=a0,
        uniform_b=b0,
        lambda_lower=lambda_l,
        lambda_upper=lambda_u,
        uniform_tau_b=tau_b0,
        uniform_tau_i=tau_i0,
        uniform_u_interface=ui0,
        uniform_w_upper=w0,
        uniform_u_upper=uu0,
        uniform_q_upper=qu0,
    )


_validate_parameters()
DERIVED = _derive_model()

# Runtime parameter vector passed explicitly to every jitted closure call.
# This avoids Numba's compile-time binding of numeric module globals.
# Indices are documented beside the construction and used only in jitted code.
_MODEL_VECTOR = np.asarray(
    [
        DERIVED.m_lower,          # 0
        DERIVED.m_upper,          # 1
        DERIVED.upper_a,          # 2
        DERIVED.upper_b,          # 3
        DERIVED.lambda_lower,     # 4
        DERIVED.lambda_upper,     # 5
        N_LOWER,                  # 6
        N_UPPER,                  # 7
        MIN_DEPTH,                # 8
        LAMBDA_MAX,               # 9
        CLOSURE_RESIDUAL_RTOL,    # 10
        CLOSURE_LAMBDA_XTOL,      # 11
        BRANCH_VELOCITY_TOL,      # 12
        float(LAMBDA_MAX_ITERATIONS),  # 13
    ],
    dtype=np.float64,
)

# Backward-compatible names used by the postprocessor and older notebooks.
M_LOWER = DERIVED.m_lower
M_UPPER = DERIVED.m_upper
UPPER_A = DERIVED.upper_a
UPPER_B = DERIVED.upper_b
LAMBDA_LOWER = DERIVED.lambda_lower
LAMBDA_UPPER = DERIVED.lambda_upper
UNIFORM_LAMBDA = DERIVED.uniform_lambda
UNIFORM_C = DERIVED.uniform_c
UNIFORM_A = DERIVED.uniform_a
UNIFORM_B = DERIVED.uniform_b
UNIFORM_TAU_B = DERIVED.uniform_tau_b
UNIFORM_TAU_I = DERIVED.uniform_tau_i
UNIFORM_U_INTERFACE = DERIVED.uniform_u_interface
UNIFORM_W_UPPER = DERIVED.uniform_w_upper
UNIFORM_U_UPPER = DERIVED.uniform_u_upper
UNIFORM_Q_UPPER = DERIVED.uniform_q_upper


# =============================================================================
# 3. STRICT LOCAL CLOSURE
# =============================================================================

# Status codes returned by the Numba kernel.
_CLOSURE_OK = 0
_CLOSURE_NONFINITE = 1
_CLOSURE_DEPTH = 2
_CLOSURE_LOWER_VELOCITY = 3
_CLOSURE_NO_POSITIVE_UPPER_BRANCH = 4
_CLOSURE_NO_BRACKET = 5
_CLOSURE_RESIDUAL = 6
_CLOSURE_SHAPE = 7

_STATUS_TEXT = {
    _CLOSURE_OK: "valid positive-shear closure",
    _CLOSURE_NONFINITE: "non-finite state or closure quantity",
    _CLOSURE_DEPTH: "non-positive or sub-minimum layer depth",
    _CLOSURE_LOWER_VELOCITY: "non-positive lower mean/interface velocity",
    _CLOSURE_NO_POSITIVE_UPPER_BRANCH: "no admissible W_u >= 0 branch",
    _CLOSURE_NO_BRACKET: "traction-continuity residual is not bracketed",
    _CLOSURE_RESIDUAL: "root residual did not satisfy tolerance",
    _CLOSURE_SHAPE: "invalid lower-profile coefficient",
}




@njit(cache=True)
def _invalid_trial(status: int) -> Tuple[float, float, float, float, float, float, float, float, float, int]:
    return (
        math.nan, math.nan, math.nan, math.nan, math.nan,
        math.nan, math.nan, math.nan, math.nan, status,
    )


@njit(cache=True)
def _invalid_closure(
    status: int,
    residual: float = math.nan,
    tau_i_upper: float = math.nan,
) -> Tuple[float, float, float, float, float, float, float, float, int, float, float]:
    return (
        math.nan, math.nan, math.nan, math.nan,
        math.nan, math.nan, math.nan, math.nan,
        status, residual, tau_i_upper,
    )


@njit(cache=True)
def _closure_state_at_lambda(
    lam: float,
    h_l: float,
    h_u: float,
    mean_l: float,
    mean_u: float,
    p: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float, float, int]:
    """Evaluate the closure at one trial lambda without clipping."""
    m_l, m_u, a_u, b_u = p[0], p[1], p[2], p[3]
    lambda_l, lambda_u = p[4], p[5]
    n_l, n_u = p[6], p[7]
    velocity_tol = p[12]

    c_l, a_l, b_l = _lower_shape_coefficients_core(lam, m_l)
    if (
        not math.isfinite(c_l)
        or not math.isfinite(a_l)
        or not math.isfinite(b_l)
        or c_l <= 0.0
        or a_l <= 0.0
        or b_l <= 0.0
    ):
        return _invalid_trial(_CLOSURE_SHAPE)

    u_interface = mean_l / a_l
    w_upper = (mean_u - u_interface) / a_u

    if not math.isfinite(u_interface) or not math.isfinite(w_upper):
        return _invalid_trial(_CLOSURE_NONFINITE)
    if u_interface <= velocity_tol:
        return _invalid_trial(_CLOSURE_LOWER_VELOCITY)

    # A tiny negative value can arise at the W=0 admissibility boundary.
    if w_upper < 0.0 and w_upper >= -velocity_tol * max(1.0, abs(mean_u)):
        w_upper = 0.0
    if w_upper < 0.0:
        return (
            math.nan,
            u_interface,
            w_upper,
            math.nan,
            math.nan,
            c_l,
            a_l,
            b_l,
            math.nan,
            _CLOSURE_NO_POSITIVE_UPPER_BRANCH,
        )

    lower_scale = u_interface / (h_l * c_l)
    upper_shear = m_u * w_upper / h_u
    if lower_scale <= 0.0 or upper_shear < 0.0:
        return _invalid_trial(_CLOSURE_NONFINITE)

    tau_b = lambda_l * lower_scale**n_l
    tau_i_upper = lambda_u * upper_shear**n_u
    residual = lam * tau_b - tau_i_upper

    momentum_l = h_l * u_interface * u_interface * b_l
    momentum_u = h_u * (
        u_interface * u_interface
        + 2.0 * a_u * u_interface * w_upper
        + b_u * w_upper * w_upper
    )

    if (
        not math.isfinite(tau_b)
        or not math.isfinite(tau_i_upper)
        or not math.isfinite(residual)
        or not math.isfinite(momentum_l)
        or not math.isfinite(momentum_u)
    ):
        return _invalid_trial(_CLOSURE_NONFINITE)

    return (
        residual,
        u_interface,
        w_upper,
        tau_b,
        tau_i_upper,
        c_l,
        a_l,
        b_l,
        momentum_l,
        _CLOSURE_OK,
    )


@njit(cache=True)
def _closure_terms_core(
    Q: np.ndarray,
    p: np.ndarray,
) -> Tuple[
    float, float, float, float, float, float, float, float, int, float, float
]:
    """Strict root solve for lambda and all local closure quantities."""
    h_l, h_u, q_l, q_u = Q[0], Q[1], Q[2], Q[3]
    min_depth, lambda_max = p[8], p[9]
    residual_rtol, lambda_xtol = p[10], p[11]
    velocity_tol = p[12]
    max_iterations = int(p[13])
    a_u, b_u = p[2], p[3]

    if (
        not math.isfinite(h_l)
        or not math.isfinite(h_u)
        or not math.isfinite(q_l)
        or not math.isfinite(q_u)
    ):
        return _invalid_closure(_CLOSURE_NONFINITE)
    if h_l <= min_depth or h_u <= min_depth:
        return _invalid_closure(_CLOSURE_DEPTH)

    mean_l = q_l / h_l
    mean_u = q_u / h_u
    if not math.isfinite(mean_l) or not math.isfinite(mean_u):
        return _invalid_closure(_CLOSURE_NONFINITE)
    if mean_l <= velocity_tol:
        return _invalid_closure(_CLOSURE_LOWER_VELOCITY)

    lo = 0.0
    hi = lambda_max

    lo_eval = _closure_state_at_lambda(lo, h_l, h_u, mean_l, mean_u, p)
    if lo_eval[9] != _CLOSURE_OK:
        return _invalid_closure(lo_eval[9])

    # Determine the largest lambda for which W_u >= 0.  The residual is only
    # physically meaningful on this positive-shear interval.
    hi_eval = _closure_state_at_lambda(hi, h_l, h_u, mean_l, mean_u, p)
    if hi_eval[9] == _CLOSURE_NO_POSITIVE_UPPER_BRANCH:
        left = lo
        right = hi
        for _ in range(max_iterations):
            mid = 0.5 * (left + right)
            mid_eval = _closure_state_at_lambda(mid, h_l, h_u, mean_l, mean_u, p)
            if mid_eval[9] == _CLOSURE_OK:
                left = mid
            else:
                right = mid
            if right - left <= lambda_xtol:
                break
        hi = left
        hi_eval = _closure_state_at_lambda(hi, h_l, h_u, mean_l, mean_u, p)

    if hi_eval[9] != _CLOSURE_OK:
        return _invalid_closure(hi_eval[9])

    r_lo = lo_eval[0]
    r_hi = hi_eval[0]
    scale_lo = max(1.0, abs(lo * lo_eval[3]), abs(lo_eval[4]))
    scale_hi = max(1.0, abs(hi * hi_eval[3]), abs(hi_eval[4]))

    if abs(r_lo) <= residual_rtol * scale_lo:
        lam = lo
        final_eval = lo_eval
    else:
        if r_lo > 0.0 or r_hi < 0.0:
            return _invalid_closure(_CLOSURE_NO_BRACKET)
        if abs(r_hi) <= residual_rtol * scale_hi:
            lam = hi
            final_eval = hi_eval
        else:
            final_eval = hi_eval
            for _ in range(max_iterations):
                mid = 0.5 * (lo + hi)
                mid_eval = _closure_state_at_lambda(
                    mid, h_l, h_u, mean_l, mean_u, p
                )
                if mid_eval[9] != _CLOSURE_OK:
                    hi = mid
                    continue
                r_mid = mid_eval[0]
                scale_mid = max(1.0, abs(mid * mid_eval[3]), abs(mid_eval[4]))
                final_eval = mid_eval
                if r_mid > 0.0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo <= lambda_xtol * max(1.0, abs(mid)):
                    break
            lam = 0.5 * (lo + hi)
            final_eval = _closure_state_at_lambda(
                lam, h_l, h_u, mean_l, mean_u, p
            )

    if final_eval[9] != _CLOSURE_OK:
        return _invalid_closure(final_eval[9])

    residual = final_eval[0]
    u_interface = final_eval[1]
    w_upper = final_eval[2]
    tau_b = final_eval[3]
    tau_i_upper = final_eval[4]
    b_l = final_eval[7]
    momentum_l = h_l * u_interface * u_interface * b_l
    momentum_u = h_u * (
        u_interface * u_interface
        + 2.0 * a_u * u_interface * w_upper
        + b_u * w_upper * w_upper
    )
    tau_i = lam * tau_b
    scale = max(1.0, abs(tau_i), abs(tau_i_upper))
    if abs(residual) > 5.0 * residual_rtol * scale:
        return _invalid_closure(_CLOSURE_RESIDUAL, residual, tau_i_upper)

    return (
        lam,
        tau_b,
        tau_i,
        u_interface,
        w_upper,
        momentum_l,
        momentum_u,
        1.0,
        _CLOSURE_OK,
        residual,
        tau_i_upper,
    )


def _closure_error_message(Q: np.ndarray, status: int, residual: float) -> str:
    state_text = np.array2string(np.asarray(Q, dtype=float), precision=12)
    reason = _STATUS_TEXT.get(int(status), f"unknown status {status}")
    return (
        f"Inadmissible two-layer closure state Q={state_text}: {reason}. "
        f"Residual={residual!r}. The model only supports positive depths and "
        "the monotone co-current branch tau_b>tau_I>=0, U_I>0, W_u>=0."
    )


def closure_diagnostics(Q: np.ndarray) -> Dict[str, float]:
    """Return closure quantities and a status code without raising."""
    q = np.ascontiguousarray(np.asarray(Q, dtype=float))
    values = _closure_terms_core(q, _MODEL_VECTOR)
    (
        lam,
        tau_b,
        tau_i,
        u_interface,
        w_upper,
        momentum_l,
        momentum_u,
        branch_flag,
        status,
        residual,
        tau_i_upper,
    ) = values
    return {
        "lambda": float(lam),
        "tau_b": float(tau_b),
        "tau_i": float(tau_i),
        "u_interface": float(u_interface),
        "w_upper": float(w_upper),
        "momentum_l": float(momentum_l),
        "momentum_u": float(momentum_u),
        "branch_flag": float(branch_flag),
        "status": int(status),
        "residual": float(residual),
        "tau_i_upper": float(tau_i_upper),
    }


def closure_terms(Q: np.ndarray) -> Tuple[
    float, float, float, float, float, float, float, float
]:
    """Strict backward-compatible closure interface.

    Returns
    -------
    lambda, tau_b, tau_I, U_I, W_u, M_l, M_u, branch_flag

    Unlike the legacy implementation, an invalid state raises ClosureError;
    no endpoint fallback or depth/velocity clipping is used.
    """
    q = np.ascontiguousarray(np.asarray(Q, dtype=float))
    values = _closure_terms_core(q, _MODEL_VECTOR)
    status = int(values[8])
    if status != _CLOSURE_OK:
        raise ClosureError(_closure_error_message(q, status, values[9]))
    return tuple(float(value) for value in values[:8])


# =============================================================================
# 4. PyPDE CALLBACKS AND CHARACTERISTIC DIAGNOSTICS
# =============================================================================


def _require_positive_depths(Q: np.ndarray) -> Tuple[float, float]:
    q = np.asarray(Q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"State must have shape (4,), received {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise ClosureError(f"State contains non-finite values: {q}.")
    h_l, h_u = float(q[0]), float(q[1])
    if h_l <= MIN_DEPTH or h_u <= MIN_DEPTH:
        raise ClosureError(
            f"Layer depths must exceed MIN_DEPTH={MIN_DEPTH:g}; "
            f"received h_l={h_l:g}, h_u={h_u:g}."
        )
    return h_l, h_u


def F(Q: np.ndarray) -> np.ndarray:
    """Numba-compatible conservative flux callback for PyPDE.

    Invalid states return NaNs instead of an endpoint/clipped closure.  The
    initial-state preflight and postprocessing diagnostics provide readable
    status information; NaNs force an inadmissible run to fail rather than
    silently changing the governing equations.
    """
    flux = np.empty(4, dtype=np.float64)
    h_l = Q[0]
    h_u = Q[1]
    if (
        not np.isfinite(h_l)
        or not np.isfinite(h_u)
        or h_l <= MIN_DEPTH
        or h_u <= MIN_DEPTH
    ):
        flux[0] = np.nan
        flux[1] = np.nan
        flux[2] = np.nan
        flux[3] = np.nan
        return flux

    closure = _closure_terms_core(Q, _MODEL_VECTOR)
    if closure[8] != _CLOSURE_OK:
        flux[0] = np.nan
        flux[1] = np.nan
        flux[2] = np.nan
        flux[3] = np.nan
        return flux

    flux[0] = Q[2]
    flux[1] = Q[3]
    flux[2] = closure[5] + 0.5 * h_l * h_l / (FR_LOWER * FR_LOWER)
    flux[3] = closure[6] + 0.5 * h_u * h_u / (FR_LOWER * FR_LOWER)
    return flux


def B(Q: np.ndarray) -> np.ndarray:
    """Numba-compatible nonconservative hydrostatic matrix callback."""
    matrix = np.zeros((4, 4), dtype=np.float64)
    h_l = Q[0]
    h_u = Q[1]
    if (
        not np.isfinite(h_l)
        or not np.isfinite(h_u)
        or h_l <= MIN_DEPTH
        or h_u <= MIN_DEPTH
    ):
        for i in range(4):
            for j in range(4):
                matrix[i, j] = np.nan
        return matrix

    matrix[2, 1] = DENSITY_RATIO * h_l / (FR_LOWER * FR_LOWER)
    matrix[3, 0] = h_u / (FR_LOWER * FR_LOWER)
    return matrix


def S(Q: np.ndarray) -> np.ndarray:
    """Numba-compatible gravity and traction source callback."""
    source = np.zeros(4, dtype=np.float64)
    h_l = Q[0]
    h_u = Q[1]
    if (
        not np.isfinite(h_l)
        or not np.isfinite(h_u)
        or h_l <= MIN_DEPTH
        or h_u <= MIN_DEPTH
    ):
        source[0] = np.nan
        source[1] = np.nan
        source[2] = np.nan
        source[3] = np.nan
        return source

    closure = _closure_terms_core(Q, _MODEL_VECTOR)
    if closure[8] != _CLOSURE_OK:
        source[0] = np.nan
        source[1] = np.nan
        source[2] = np.nan
        source[3] = np.nan
        return source

    tau_b = closure[1]
    tau_i = closure[2]
    source[2] = h_l / (FR_LOWER * FR_LOWER) + tau_i - tau_b
    source[3] = h_u / (FR_LOWER * FR_LOWER) - tau_i / DENSITY_RATIO
    return source


def checked_callbacks(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate F, B and S and raise a readable error if any are non-finite."""
    q = np.asarray(Q, dtype=float)
    diagnostic = closure_diagnostics(q)
    if diagnostic["status"] != _CLOSURE_OK:
        raise ClosureError(
            _closure_error_message(q, diagnostic["status"], diagnostic["residual"])
        )
    flux = F(q)
    matrix = B(q)
    source = S(q)
    if not (np.all(np.isfinite(flux)) and np.all(np.isfinite(matrix)) and np.all(np.isfinite(source))):
        raise ClosureError(f"Non-finite callback result at Q={q}.")
    return flux, matrix, source


def numerical_jacobian(
    function,
    state: np.ndarray,
    relative_step: float = 2.0e-6,
) -> np.ndarray:
    """Central-difference Jacobian used for preflight and eigenmode setup."""
    q = np.asarray(state, dtype=float)
    f0 = np.asarray(function(q), dtype=float)
    jac = np.empty((f0.size, q.size), dtype=float)
    for j in range(q.size):
        step = relative_step * max(1.0, abs(q[j]))
        dq = np.zeros_like(q)
        dq[j] = step
        jac[:, j] = (
            np.asarray(function(q + dq), dtype=float)
            - np.asarray(function(q - dq), dtype=float)
        ) / (2.0 * step)
    return jac


def pypde_like_flux_jacobian(state: np.ndarray) -> np.ndarray:
    """Approximate dF/dQ with the forward steps used by PyPDE's backend."""
    q = np.asarray(state, dtype=float)
    f0 = F(q)
    jac = np.empty((4, 4), dtype=float)
    root_eps = math.sqrt(np.finfo(float).eps)
    for j in range(4):
        step = max(root_eps * abs(q[j]), root_eps)
        qp = q.copy()
        qp[j] += step
        jac[:, j] = (F(qp) - f0) / step
    return jac


def quasilinear_matrix(state: np.ndarray, *, pypde_like: bool = True) -> np.ndarray:
    """Return A=dF/dQ+B at one state."""
    jac = pypde_like_flux_jacobian(state) if pypde_like else numerical_jacobian(F, state)
    return jac + B(state)


def characteristic_speeds(
    state: np.ndarray,
    *,
    imag_tolerance: float = HYPERBOLICITY_IMAG_TOL,
    raise_on_complex: bool = True,
) -> np.ndarray:
    """Return local characteristic eigenvalues and optionally enforce hyperbolicity."""
    values = np.linalg.eigvals(quasilinear_matrix(state))
    max_imag = float(np.max(np.abs(values.imag)))
    if raise_on_complex and max_imag > imag_tolerance:
        raise RuntimeError(
            f"Loss of numerical hyperbolicity at Q={state}: "
            f"max |Im(lambda_A)|={max_imag:.3e}."
        )
    order = np.argsort(values.real)
    return values[order]


# =============================================================================
# 5. UNIFORM STATE AND INITIAL CONDITIONS
# =============================================================================


def uniform_state() -> np.ndarray:
    return np.asarray([1.0, DEPTH_RATIO, 1.0, UNIFORM_Q_UPPER], dtype=float)


def local_normal_flow_fluxes(h_l: float, h_u: float) -> Tuple[float, float]:
    """Legacy local-normal-flow discharge reconstruction for comparison."""
    h_l = float(h_l)
    h_u = float(h_u)
    if h_l <= MIN_DEPTH or h_u <= MIN_DEPTH:
        raise ValueError("Local normal-flow depths must exceed MIN_DEPTH.")

    tau_b = (h_l + DENSITY_RATIO * h_u) / FR_LOWER**2
    tau_i = DENSITY_RATIO * h_u / FR_LOWER**2
    lam = tau_i / tau_b
    c_l, a_l, _ = lower_shape_coefficients(lam)
    u_interface = h_l * (tau_b / LAMBDA_LOWER) ** (1.0 / N_LOWER) * c_l
    w_upper = h_u / M_UPPER * (tau_i / LAMBDA_UPPER) ** (1.0 / N_UPPER)
    q_l = h_l * u_interface * a_l
    q_u = h_u * (u_interface + UPPER_A * w_upper)
    return float(q_l), float(q_u)


def output_times() -> np.ndarray:
    count = int(round(FINAL_TIME / OUTPUT_INTERVAL))
    return OUTPUT_INTERVAL * np.arange(1, count + 1, dtype=float)


def _periodic_distance(x: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(np.asarray(x, dtype=float) - center)
    return np.minimum(delta, DOMAIN_LENGTH - delta)


def disturbance_envelope(x: np.ndarray) -> np.ndarray:
    if PERTURBATION_ENVELOPE == "periodic":
        return np.ones_like(np.asarray(x, dtype=float))
    distance = _periodic_distance(np.asarray(x, dtype=float), PACKET_CENTER)
    return np.exp(-(distance / PACKET_WIDTH) ** 2)


def linearized_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """Return A0=dF/dQ+B and C0=dS/dQ at the normal flow."""
    q0 = uniform_state()
    a0 = numerical_jacobian(F, q0) + B(q0)
    c0 = numerical_jacobian(S, q0)
    return a0, c0


def _normalize_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=complex).copy()
    for j in range(result.shape[1]):
        norm = np.linalg.norm(result[:, j])
        if norm == 0.0:
            raise RuntimeError("Zero eigenvector encountered during branch tracking.")
        result[:, j] /= norm
    return result


def _best_permutation(
    previous_values: np.ndarray,
    previous_vectors: np.ndarray,
    new_values: np.ndarray,
    new_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign four roots globally using eigenvector overlap and eigenvalue continuity."""
    best_cost = math.inf
    best_perm: Optional[Tuple[int, ...]] = None
    scale = max(1.0, float(np.max(np.abs(previous_values))))
    for perm in permutations(range(4)):
        cost = 0.0
        for old_index, new_index in enumerate(perm):
            overlap = abs(np.vdot(previous_vectors[:, old_index], new_vectors[:, new_index]))
            eig_distance = abs(new_values[new_index] - previous_values[old_index]) / scale
            cost += (1.0 - min(1.0, overlap)) + 0.15 * eig_distance
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Unable to match eigenvalue branches.")
    indices = np.asarray(best_perm, dtype=int)
    return new_values[indices], new_vectors[:, indices]


def linear_mode(
    wavenumber: float,
    branch: str,
    continuation_points: int = 96,
) -> Tuple[complex, np.ndarray]:
    """Return a tracked hydrodynamic eigenvalue and right eigenvector.

    The eigenproblem is

        sigma v = (C0 - i*k*A0) v.

    The two roots tending to zero as k->0 are labelled by their depth shape:
    the branch with the larger free-surface elevation amplitude is labelled
    ``free_surface`` and the other ``interface``.  That label is then continued
    to the requested wavenumber by eigenvector overlap.
    """
    if branch not in _ALLOWED_BRANCHES:
        raise ValueError(f"branch must be one of {sorted(_ALLOWED_BRANCHES)}.")
    if not np.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be finite and positive.")

    a0, c0 = linearized_matrices()
    k_start = min(1.0e-5, wavenumber * 1.0e-4)
    if wavenumber <= k_start * (1.0 + 1.0e-12):
        k_values = np.asarray([wavenumber])
    else:
        k_values = np.geomspace(k_start, wavenumber, continuation_points)

    values, vectors = np.linalg.eig(c0 - 1j * k_values[0] * a0)
    vectors = _normalize_eigenvectors(vectors)

    hydro = np.argsort(np.abs(values))[:2]
    surface_scores = []
    for index in hydro:
        depth_norm = math.sqrt(abs(vectors[0, index]) ** 2 + abs(vectors[1, index]) ** 2)
        surface_scores.append(abs(vectors[0, index] + vectors[1, index]) / max(depth_norm, 1.0e-30))
    free_index = hydro[int(np.argmax(surface_scores))]
    interface_index = hydro[1 - int(np.argmax(surface_scores))]

    remaining = [i for i in range(4) if i not in {free_index, interface_index}]
    remaining.sort(key=lambda i: values[i].real)
    ordering = [interface_index, free_index] + remaining
    values = values[ordering]
    vectors = vectors[:, ordering]

    for k in k_values[1:]:
        new_values, new_vectors = np.linalg.eig(c0 - 1j * k * a0)
        new_vectors = _normalize_eigenvectors(new_vectors)
        values, vectors = _best_permutation(
            values, vectors, new_values, new_vectors
        )

    index = 1 if branch == "free_surface" else 0
    eigenvalue = complex(values[index])
    eigenvector = np.asarray(vectors[:, index], dtype=complex)

    reference = eigenvector[0] + eigenvector[1] if branch == "free_surface" else eigenvector[0]
    if abs(reference) < 1.0e-14:
        raise RuntimeError(f"Cannot normalize {branch} eigenmode: reference amplitude is zero.")
    eigenvector *= np.exp(-1j * np.angle(reference)) / abs(reference)
    return eigenvalue, eigenvector


def _legacy_periodic_depths(x: np.ndarray, amplitude: float) -> Tuple[np.ndarray, np.ndarray]:
    phase = 2.0 * np.pi * np.asarray(x, dtype=float) / PERTURBATION_WAVELENGTH
    envelope = disturbance_envelope(x)
    disturbance_l = envelope * np.sin(phase)
    disturbance_u = envelope * np.sin(phase + UPPER_LAYER_PHASE_SHIFT)
    h_l = 1.0 + amplitude * disturbance_l
    h_u = DEPTH_RATIO * (1.0 + amplitude * disturbance_u)
    return h_l, h_u


def initial_condition(
    x: np.ndarray,
    *,
    mode: Optional[str] = None,
    amplitude: Optional[float] = None,
    branch: Optional[str] = None,
) -> np.ndarray:
    """Construct a smooth cell-centred initial state.

    Optional arguments are used by the comparison script and do not alter the
    module-level defaults.
    """
    x = np.asarray(x, dtype=float)
    selected_mode = INITIAL_CONDITION_MODE if mode is None else mode
    selected_amplitude = PERTURBATION_AMPLITUDE if amplitude is None else float(amplitude)
    selected_branch = EIGENMODE_BRANCH if branch is None else branch

    if selected_mode not in _ALLOWED_IC_MODES:
        raise ValueError(f"mode must be one of {sorted(_ALLOWED_IC_MODES)}.")
    if selected_amplitude < 0.0 or not np.isfinite(selected_amplitude):
        raise ValueError("amplitude must be finite and non-negative.")

    q0 = uniform_state()
    Q = np.tile(q0, (x.size, 1)).astype(float)

    if selected_mode == "linear_eigenmode":
        k = 2.0 * np.pi / PERTURBATION_WAVELENGTH
        _, vector = linear_mode(k, selected_branch)
        phase = np.exp(1j * k * x)
        envelope = disturbance_envelope(x)
        perturbation = selected_amplitude * envelope[:, None] * np.real(
            phase[:, None] * vector[None, :]
        )
        Q += perturbation

    elif selected_mode == "depth_only":
        h_l, h_u = _legacy_periodic_depths(x, selected_amplitude)
        Q[:, 0] = h_l
        Q[:, 1] = h_u

    else:
        h_l, h_u = _legacy_periodic_depths(x, selected_amplitude)
        Q[:, 0] = h_l
        Q[:, 1] = h_u
        for i in range(x.size):
            if selected_mode == "legacy_local_normal_flow":
                q_l, q_u = local_normal_flow_fluxes(h_l[i], h_u[i])
            else:
                # Kept only for exact reproduction of the previous option.
                q_l = h_l[i] ** 1.5
                q_u = UNIFORM_Q_UPPER * (h_u[i] / DEPTH_RATIO) ** 1.5
            Q[i, 2] = q_l
            Q[i, 3] = q_u

    if np.min(Q[:, 0]) <= MIN_DEPTH or np.min(Q[:, 1]) <= MIN_DEPTH:
        raise ValueError(
            "The selected perturbation produces a non-positive layer depth. "
            "Reduce PERTURBATION_AMPLITUDE."
        )
    return np.ascontiguousarray(Q)


def preflight_initial_state(x: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """Check closure, source balance scale, and hyperbolicity before the run."""
    indices: Iterable[int]
    if PREFLIGHT_EVERY_CELL:
        indices = range(Q.shape[0])
    else:
        indices = np.unique(np.linspace(0, Q.shape[0] - 1, min(32, Q.shape[0])).astype(int))

    max_residual = 0.0
    max_imag = 0.0
    min_w = math.inf
    lambda_min = math.inf
    lambda_max = -math.inf
    for i in indices:
        diagnostic = closure_diagnostics(Q[i])
        if diagnostic["status"] != _CLOSURE_OK:
            raise ClosureError(_closure_error_message(Q[i], diagnostic["status"], diagnostic["residual"]))
        max_residual = max(max_residual, abs(diagnostic["residual"]))
        min_w = min(min_w, diagnostic["w_upper"])
        lambda_min = min(lambda_min, diagnostic["lambda"])
        lambda_max = max(lambda_max, diagnostic["lambda"])
        eigenvalues = characteristic_speeds(Q[i], raise_on_complex=False)
        max_imag = max(max_imag, float(np.max(np.abs(eigenvalues.imag))))

    if max_imag > HYPERBOLICITY_IMAG_TOL:
        raise RuntimeError(
            f"Initial state is not numerically hyperbolic: max imaginary "
            f"characteristic part={max_imag:.3e}."
        )
    return {
        "closure_residual_max": max_residual,
        "characteristic_imag_max": max_imag,
        "w_upper_min": min_w,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "h_lower_min": float(np.min(Q[:, 0])),
        "h_upper_min": float(np.min(Q[:, 1])),
    }


# =============================================================================
# 6. PYpDE RUNNER
# =============================================================================


def _single_pypde_call(
    pde_solver,
    initial_state: np.ndarray,
    duration: float,
    number_of_outputs: int,
) -> np.ndarray:
    """Run PyPDE on a private mutable copy of a state."""
    work_state = np.array(initial_state, dtype=float, order="C", copy=True)
    return np.asarray(
        pde_solver(
            work_state,
            duration,
            [DOMAIN_LENGTH],
            F=F,
            B=B,
            S=S,
            boundaryTypes=BOUNDARY_TYPE,
            cfl=CFL,
            order=RECONSTRUCTION_ORDER,
            ndt=number_of_outputs,
            flux=NUMERICAL_FLUX,
            stiff=STIFF_SOURCE,
            nThreads=N_THREADS,
        ),
        dtype=float,
    )


def run_pypde(pde_solver, initial_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run PyPDE with nominal-uniform or exact-segmented output control."""
    times = output_times()
    if OUTPUT_MODE == "uniform_nominal":
        out = _single_pypde_call(
            pde_solver,
            initial_state,
            FINAL_TIME,
            len(times),
        )
        return out, times

    states = []
    current_state = np.array(initial_state, dtype=float, copy=True, order="C")
    current_time = 0.0
    for target_time in times:
        duration = float(target_time - current_time)
        segment = _single_pypde_call(pde_solver, current_state, duration, 1)
        current_state = np.array(segment[-1], dtype=float, copy=True, order="C")
        states.append(current_state)
        current_time = float(target_time)
    return np.stack(states, axis=0), times


def print_parameter_summary(preflight: Optional[Dict[str, float]] = None) -> None:
    qn = uniform_state()
    closure = closure_diagnostics(qn)
    speeds = characteristic_speeds(qn)
    print("\nAudited two-layer power-law Kármán–Pohlhausen model")
    print("----------------------------------------------------------------")
    print(f"Fr_l                              = {FR_LOWER:.10g}")
    print(f"h_r                               = {DEPTH_RATIO:.10g}")
    print(f"rho_r                             = {DENSITY_RATIO:.10g}")
    print(f"n_l, n_u                          = {N_LOWER:.10g}, {N_UPPER:.10g}")
    print(f"scaled consistency ratio          = {SCALED_CONSISTENCY_RATIO:.10g}")
    print(f"Lambda_l, Lambda_u                = {LAMBDA_LOWER:.10g}, {LAMBDA_UPPER:.10g}")
    print(f"normal-flow lambda                = {UNIFORM_LAMBDA:.10g}")
    print(f"normal-flow q_u                   = {UNIFORM_Q_UPPER:.10g}")
    print(f"normal-flow source residual norm  = {np.linalg.norm(S(qn)):.3e}")
    print(f"normal-flow closure residual      = {closure['residual']:.3e}")
    print("normal-flow characteristic speeds = " + ", ".join(f"{z.real:.9g}" for z in speeds))
    print(f"initial-condition mode            = {INITIAL_CONDITION_MODE}")
    if INITIAL_CONDITION_MODE == "linear_eigenmode":
        print(f"eigenmode branch                  = {EIGENMODE_BRANCH}")
    print(f"perturbation amplitude            = {PERTURBATION_AMPLITUDE:.6g}")
    print(f"output mode / interval            = {OUTPUT_MODE} / {OUTPUT_INTERVAL:g}")
    if preflight is not None:
        print(f"initial lambda range              = [{preflight['lambda_min']:.8g}, {preflight['lambda_max']:.8g}]")
        print(f"initial min W_u                   = {preflight['w_upper_min']:.3e}")
        print(f"initial max closure residual      = {preflight['closure_residual_max']:.3e}")
        print(f"initial max characteristic Im     = {preflight['characteristic_imag_max']:.3e}")
    print("----------------------------------------------------------------\n")


def main() -> None:
    try:
        from pypde import pde_solver
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyPDE is not installed in this Python environment. Install the "
            "Haran Jackson PyPDE package in a supported environment, then rerun."
        ) from exc

    dx = DOMAIN_LENGTH / NX
    x = (np.arange(NX, dtype=float) + 0.5) * dx
    Q_initial = np.array(initial_condition(x), dtype=float, copy=True, order="C")
    preflight = preflight_initial_state(x, Q_initial)
    print_parameter_summary(preflight)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_x.npy", x)
    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_Q0.npy", Q_initial)

    out, positive_times = run_pypde(pde_solver, Q_initial)
    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_out.npy", out)
    np.save(OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_output_times.npy", positive_times)

    if RUN_POSTPROCESSING:
        from two_layer_powerlaw_postprocess import postprocess_solution

        postprocess_solution(
            x=x,
            initial_state=Q_initial,
            out=out,
            output_times=positive_times,
            output_directory=OUTPUT_DIRECTORY,
            output_stem=OUTPUT_STEM,
        )


if __name__ == "__main__":
    main()
