#!/usr/bin/env python3
"""Parameter sweep for steady-uniform Liu-Mei-Exner eigenvalues.

The script does not run PyPDE.  For each selected parameter value it:

1. constructs the steady-uniform state U = [h, q, u_p, b],
2. forms A(U) = dF/dU + B(U) by central finite differences,
3. computes the four eigenvalues of A,
4. writes a human-readable text table.

Keep this file beside:
    bingham_roll_wave_erodible_bed_pypde.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path
import math

import numpy as np

import bingham_roll_wave_erodible_bed_pypde as model


# =============================================================================
# 1. User settings
# =============================================================================

# Examples: "alpha", "beta", "bed_slope", "theta_crit",
#           "bed_porosity", "d_rel", "density_ratio", "d90_d30".
SWEEP_PARAMETER = "alpha"
SWEEP_VALUES = np.linspace(0.10, 0.50, 81, dtype=float)

# False: fixed-bed Liu-Mei hydrodynamics plus a stationary bed mode.
# True : full Liu-Mei-Exner matrix with Rickenmann-Exner coupling.
USE_MOBILE_BED = True

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / f"eigenvalues_vs_{SWEEP_PARAMETER}.txt"
)
RELATIVE_STEP = 1.0e-6
IMAGINARY_TOLERANCE = 1.0e-9

# Keep eigenvalue labels continuous from one parameter value to the next.
# This is more useful for plotting than sorting independently at every row.
TRACK_EIGENVALUE_BRANCHES = True


# =============================================================================
# 2. Parameter container
# =============================================================================

@dataclass(frozen=True)
class Parameters:
    alpha: float = model.ALPHA
    beta: float = model.BETA
    bed_slope: float = model.BED_SLOPE

    bed_porosity: float = model.BED_POROSITY
    density_ratio: float = model.DENSITY_RATIO
    d_rel: float = model.D_REL
    d90_d30: float = model.D90_D30
    theta_crit: float = model.THETA_CRIT

    depth_floor: float = model.DEPTH_FLOOR
    velocity_floor: float = model.VELOCITY_FLOOR
    layer_floor: float = model.LAYER_FLOOR
    delta_h0: float = model.DELTA_H0
    sign_eps: float = model.SIGN_EPS
    theta_smooth: float = model.THETA_SMOOTH


BASE_PARAMETERS = Parameters()


# =============================================================================
# 3. Model functions with parameters passed explicitly
# =============================================================================

def validate_parameters(p: Parameters) -> None:
    if not 0.0 < p.alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if p.beta <= 0.0:
        raise ValueError("beta must be positive.")
    if p.bed_slope <= 0.0:
        raise ValueError("bed_slope must be positive.")
    if not 0.0 <= p.bed_porosity < 1.0:
        raise ValueError("bed_porosity must lie in [0, 1).")
    if p.density_ratio <= 1.0:
        raise ValueError("density_ratio must exceed 1.")
    if p.d_rel <= 0.0 or p.d90_d30 <= 0.0:
        raise ValueError("d_rel and d90_d30 must be positive.")


def uniform_state(p: Parameters) -> np.ndarray:
    """Exact dimensionless Liu-Mei steady-uniform state."""
    h = 1.0
    h0 = 1.0 - p.alpha
    up = 0.5 * (1.0 - p.alpha) ** 2
    q = up * (h - h0 / 3.0)
    b = 0.0
    return np.array([h, q, up, b], dtype=float)


def smooth_sign(value: float, epsilon: float) -> float:
    return value / math.sqrt(value * value + epsilon * epsilon)


def bounded_inverse(depth: float, delta: float) -> float:
    """Return [1-exp(-depth/delta)]/depth with a finite zero limit."""
    if depth <= 1.0e-8 * delta:
        return 1.0 / delta
    return -math.expm1(-depth / delta) / depth


def positive_part(value: float, epsilon: float) -> float:
    return 0.5 * (value + math.sqrt(value * value + epsilon * epsilon))


def closure_layers(U: np.ndarray, p: Parameters):
    """Recover regularized closure variables exactly as in the solver."""
    h = max(U[0], 2.0 * p.layer_floor, p.depth_floor)
    q = U[1]
    up = U[2]

    if abs(up) < p.velocity_floor:
        up = p.velocity_floor if up >= 0.0 else -p.velocity_floor

    h0 = 3.0 * (h - q / up)
    h0 = min(max(h0, p.layer_floor), h - p.layer_floor)
    hp = h - h0
    return h, q, up, h0, hp


def basal_stress_hat(U: np.ndarray, p: Parameters) -> float:
    _, _, up, h0, _ = closure_layers(U, p)
    return (
        p.alpha * smooth_sign(up, p.sign_eps)
        + 2.0 * up * bounded_inverse(h0, p.delta_h0)
    )


def shields_parameter(U: np.ndarray, p: Parameters) -> float:
    theta = math.atan(p.bed_slope)
    denominator = (p.density_ratio - 1.0) * p.d_rel
    return abs(basal_stress_hat(U, p)) * math.sin(theta) / denominator


def bedload_hat(U: np.ndarray, p: Parameters) -> float:
    """Dimensionless Rickenmann discharge used by the Exner flux."""
    if not USE_MOBILE_BED:
        return 0.0

    h, q, up, _, _ = closure_layers(U, p)
    theta = math.atan(p.bed_slope)
    theta_b = shields_parameter(U, p)
    excess = positive_part(theta_b - p.theta_crit, p.theta_smooth)

    coefficient = (
        3.1
        * (p.beta * math.cos(theta)) ** 0.05
        * p.d_rel**1.5
        * p.d90_d30**0.2
    )
    local_froude_factor = abs(q) / h**1.5

    return (
        smooth_sign(up, p.sign_eps)
        * coefficient
        * math.sqrt(theta_b)
        * excess
        * local_froude_factor**1.1
    )


def flux(U: np.ndarray, p: Parameters) -> np.ndarray:
    h, q, up, b = U

    F = np.empty(4, dtype=float)
    F[0] = q
    F[1] = (
        7.0 * q * up / 5.0
        - 2.0 * h * up * up / 5.0
        + h * h / (2.0 * p.beta)
    )
    F[2] = 0.5 * up * up + (h + b) / p.beta
    F[3] = bedload_hat(U, p) / (1.0 - p.bed_porosity)
    return F


def nonconservative(U: np.ndarray, p: Parameters) -> np.ndarray:
    B = np.zeros((4, 4), dtype=float)
    B[1, 3] = U[0] / p.beta
    return B


def flux_jacobian(U: np.ndarray, p: Parameters) -> np.ndarray:
    """Central-difference approximation of dF/dU."""
    U = np.asarray(U, dtype=float)
    nvar = U.size
    J = np.empty((nvar, nvar), dtype=float)

    for j in range(nvar):
        step = RELATIVE_STEP * max(1.0, abs(U[j]))
        dU = np.zeros(nvar, dtype=float)
        dU[j] = step
        J[:, j] = (flux(U + dU, p) - flux(U - dU, p)) / (2.0 * step)

    return J


def system_matrix(U: np.ndarray, p: Parameters) -> np.ndarray:
    return flux_jacobian(U, p) + nonconservative(U, p)


# =============================================================================
# 4. Eigenvalue ordering and output
# =============================================================================

def initial_eigenvalue_order(values: np.ndarray) -> np.ndarray:
    """Sort first row by real part, then by imaginary part."""
    order = np.lexsort((values.imag, values.real))
    return values[order]


def continue_eigenvalue_branches(
    previous: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Choose the permutation nearest to the previous eigenvalue vector."""
    best_values = None
    best_cost = math.inf

    for permutation in permutations(range(current.size)):
        candidate = current[list(permutation)]
        cost = float(np.sum(np.abs(candidate - previous) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_values = candidate

    return best_values


def sweep() -> tuple[np.ndarray, list[str]]:
    if not hasattr(BASE_PARAMETERS, SWEEP_PARAMETER):
        raise ValueError(
            f"Unknown SWEEP_PARAMETER={SWEEP_PARAMETER!r}. "
            f"Choose a field of Parameters."
        )

    values = np.asarray(SWEEP_VALUES, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("SWEEP_VALUES must be a non-empty one-dimensional array.")

    rows = []
    previous_eigenvalues = None

    for parameter_value in values:
        p = replace(
            BASE_PARAMETERS,
            **{SWEEP_PARAMETER: float(parameter_value)},
        )
        validate_parameters(p)

        U0 = uniform_state(p)
        h0 = 1.0 - p.alpha
        hp = p.alpha

        A = system_matrix(U0, p)
        eigenvalues = np.linalg.eigvals(A)

        if previous_eigenvalues is None:
            eigenvalues = initial_eigenvalue_order(eigenvalues)
        elif TRACK_EIGENVALUE_BRANCHES:
            eigenvalues = continue_eigenvalue_branches(
                previous_eigenvalues,
                eigenvalues,
            )
        else:
            eigenvalues = initial_eigenvalue_order(eigenvalues)

        previous_eigenvalues = eigenvalues.copy()

        max_imaginary = float(np.max(np.abs(eigenvalues.imag)))
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        hyperbolic = 1.0 if max_imaginary <= IMAGINARY_TOLERANCE else 0.0

        row = [
            parameter_value,
            U0[0],
            U0[1],
            U0[2],
            U0[3],
            h0,
            hp,
        ]

        for eigenvalue in eigenvalues:
            row.extend([eigenvalue.real, eigenvalue.imag])

        row.extend([max_imaginary, spectral_radius, hyperbolic])
        rows.append(row)

    columns = [
        SWEEP_PARAMETER,
        "h",
        "q",
        "u_p",
        "b",
        "h0",
        "h_p",
    ]

    for i in range(1, 5):
        columns.extend([f"lambda{i}_real", f"lambda{i}_imag"])

    columns.extend([
        "max_abs_imaginary",
        "spectral_radius",
        "hyperbolic_flag",
    ])

    return np.asarray(rows, dtype=float), columns


def write_text_file(data: np.ndarray, columns: list[str]) -> None:
    p = BASE_PARAMETERS
    header = "\n".join(
        [
            "Steady-uniform Liu-Mei-Exner characteristic eigenvalues",
            f"sweep_parameter = {SWEEP_PARAMETER}",
            f"mobile_bed = {USE_MOBILE_BED}",
            f"branch_tracking = {TRACK_EIGENVALUE_BRANCHES}",
            f"relative_jacobian_step = {RELATIVE_STEP:.16e}",
            f"imaginary_tolerance = {IMAGINARY_TOLERANCE:.16e}",
            "Base parameters (the swept field is overridden row by row):",
            f"alpha = {p.alpha:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "alpha" else ""),
            f"beta = {p.beta:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "beta" else ""),
            f"bed_slope = {p.bed_slope:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "bed_slope" else ""),
            f"bed_porosity = {p.bed_porosity:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "bed_porosity" else ""),
            f"density_ratio = {p.density_ratio:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "density_ratio" else ""),
            f"d_rel = {p.d_rel:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "d_rel" else ""),
            f"d90_d30 = {p.d90_d30:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "d90_d30" else ""),
            f"theta_crit = {p.theta_crit:.16e}" + ("  [swept]" if SWEEP_PARAMETER == "theta_crit" else ""),
            "Columns:",
            " ".join(columns),
            "hyperbolic_flag: 1 = all |Im(lambda)| <= tolerance; 0 = otherwise",
        ]
    )

    np.savetxt(
        OUTPUT_FILE,
        data,
        fmt="%.12e",
        delimiter="\t",
        header=header,
    )


def main() -> None:
    data, columns = sweep()
    write_text_file(data, columns)

    print(f"Wrote {data.shape[0]} parameter states to: {OUTPUT_FILE.resolve()}")
    print(f"Sweep: {SWEEP_PARAMETER} from {data[0, 0]:.6g} to {data[-1, 0]:.6g}")
    print(f"Mobile bed: {USE_MOBILE_BED}")
    print(
        "Non-hyperbolic rows: "
        f"{int(np.count_nonzero(data[:, -1] == 0.0))}"
    )
    print("Column names:")
    print("  " + "\n  ".join(columns))


if __name__ == "__main__":
    main()
