#!/usr/bin/env python3
"""Postprocessing for the audited two-layer power-law PyPDE model.

The caller supplies the true protected initial state and the explicit positive
output times.  No time array is reconstructed from frame counts, and no evolved
work buffer is relabelled as t=0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import two_layer_powerlaw_kp_pypde as model


WRITE_ALL_TEXT_FRAMES = True
NUMBER_OF_PLOTTED_FRAMES: Optional[int] = None  # None writes every frame
PLOT_DPI = 240
SHOW_PLOTS = False
HYPERBOLICITY_CELL_STRIDE = 5


def _safe_mean(discharge: np.ndarray, depth: np.ndarray) -> np.ndarray:
    result = np.full_like(discharge, np.nan, dtype=float)
    valid = np.isfinite(depth) & (depth > model.MIN_DEPTH)
    result[valid] = discharge[valid] / depth[valid]
    return result


def recover_frame(Q: np.ndarray) -> Dict[str, np.ndarray]:
    """Recover local closure fields without hiding invalid states."""
    Q = np.asarray(Q, dtype=float)
    nx = Q.shape[0]

    h_l = Q[:, 0].copy()
    h_u = Q[:, 1].copy()
    q_l = Q[:, 2].copy()
    q_u = Q[:, 3].copy()
    mean_l = _safe_mean(q_l, h_l)
    mean_u = _safe_mean(q_u, h_u)

    names = [
        "lambda",
        "tau_b",
        "tau_i",
        "u_interface",
        "w_upper",
        "momentum_l",
        "momentum_u",
        "branch_flag",
        "status",
        "residual",
        "tau_i_upper",
    ]
    closure = {name: np.full(nx, np.nan, dtype=float) for name in names}

    for i in range(nx):
        values = model.closure_diagnostics(Q[i])
        for name in names:
            closure[name][i] = values[name]

    u_surface = closure["u_interface"] + closure["w_upper"]
    traction_mismatch = closure["tau_i"] - closure["tau_i_upper"]

    return {
        "h_l": h_l,
        "h_u": h_u,
        "eta": h_l + h_u,
        "q_l": q_l,
        "q_u": q_u,
        "mean_l": mean_l,
        "mean_u": mean_u,
        "u_interface": closure["u_interface"],
        "u_surface": u_surface,
        "w_upper": closure["w_upper"],
        "lambda": closure["lambda"],
        "tau_b": closure["tau_b"],
        "tau_i": closure["tau_i"],
        "tau_i_upper": closure["tau_i_upper"],
        "traction_mismatch": traction_mismatch,
        "momentum_l": closure["momentum_l"],
        "momentum_u": closure["momentum_u"],
        "branch_flag": closure["branch_flag"],
        "closure_status": closure["status"],
        "closure_residual": closure["residual"],
    }


def _frame_table(x: np.ndarray, fields: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    names = [
        "x",
        "h_lower",
        "h_upper",
        "free_surface",
        "q_lower",
        "q_upper",
        "mean_u_lower",
        "mean_u_upper",
        "u_interface",
        "u_surface",
        "upper_velocity_increment",
        "stress_ratio_lambda",
        "tau_b",
        "tau_I_lower_definition",
        "tau_I_upper_rheology",
        "traction_mismatch",
        "momentum_lower",
        "momentum_upper",
        "closure_branch_flag",
        "closure_status_code",
        "closure_residual",
    ]
    table = np.column_stack(
        [
            x,
            fields["h_l"],
            fields["h_u"],
            fields["eta"],
            fields["q_l"],
            fields["q_u"],
            fields["mean_l"],
            fields["mean_u"],
            fields["u_interface"],
            fields["u_surface"],
            fields["w_upper"],
            fields["lambda"],
            fields["tau_b"],
            fields["tau_i"],
            fields["tau_i_upper"],
            fields["traction_mismatch"],
            fields["momentum_l"],
            fields["momentum_u"],
            fields["branch_flag"],
            fields["closure_status"],
            fields["closure_residual"],
        ]
    )
    return table, "\t".join(names)


def write_snapshot(
    x: np.ndarray,
    fields: Dict[str, np.ndarray],
    filename: Path,
) -> None:
    table, header = _frame_table(x, fields)
    np.savetxt(
        filename,
        table,
        delimiter="\t",
        header=header,
        comments="",
        fmt="%.12e",
    )


def plot_snapshot(
    x: np.ndarray,
    fields: Dict[str, np.ndarray],
    time: float,
    filename: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    axes[0, 0].plot(x, fields["h_l"], label=r"$h_\ell$")
    axes[0, 0].plot(x, fields["eta"], label=r"$h_\ell+h_u$")
    axes[0, 0].set_xlabel(r"$x$")
    axes[0, 0].set_ylabel("interface / free-surface elevation")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(x, fields["mean_l"], label=r"$\bar u_\ell$")
    axes[0, 1].plot(x, fields["mean_u"], label=r"$\bar u_u$")
    axes[0, 1].plot(x, fields["u_interface"], label=r"$U_I$")
    axes[0, 1].plot(x, fields["u_surface"], label=r"$U_s$")
    axes[0, 1].set_xlabel(r"$x$")
    axes[0, 1].set_ylabel("velocity")
    axes[0, 1].legend(ncol=2)
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(x, fields["tau_b"], label=r"$\tau_b$")
    axes[1, 0].plot(x, fields["tau_i"], label=r"$\tau_I$")
    axes[1, 0].set_xlabel(r"$x$")
    axes[1, 0].set_ylabel("dimensionless traction")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(x, fields["lambda"], label=r"$\lambda=\tau_I/\tau_b$")
    axes[1, 1].plot(x, fields["branch_flag"], label="valid closure")
    axes[1, 1].set_xlabel(r"$x$")
    axes[1, 1].set_ylabel("closure diagnostic")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"Two-layer power-law model, dimensionless time $t={time:.8g}$")
    fig.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _select_plot_indices(number_of_frames: int) -> np.ndarray:
    if NUMBER_OF_PLOTTED_FRAMES is None:
        return np.arange(number_of_frames, dtype=int)
    count = min(int(NUMBER_OF_PLOTTED_FRAMES), number_of_frames)
    return np.unique(np.rint(np.linspace(0, number_of_frames - 1, count)).astype(int))


def _harmonic_amplitude(x: np.ndarray, signal: np.ndarray) -> float:
    valid = np.isfinite(signal)
    if np.count_nonzero(valid) < 4:
        return np.nan
    xv = x[valid]
    yv = signal[valid] - np.mean(signal[valid])
    k = 2.0 * np.pi / model.PERTURBATION_WAVELENGTH
    coefficient = np.sum(yv * np.exp(-1j * k * xv)) / yv.size
    return float(2.0 * abs(coefficient))


def _hyperbolicity_max_imag(Q: np.ndarray) -> float:
    maximum = 0.0
    for i in range(0, Q.shape[0], max(1, HYPERBOLICITY_CELL_STRIDE)):
        try:
            eig = model.characteristic_speeds(Q[i], raise_on_complex=False)
        except Exception:
            return np.inf
        maximum = max(maximum, float(np.max(np.abs(eig.imag))))
    return maximum


def postprocess_solution(
    *,
    x: np.ndarray,
    initial_state: np.ndarray,
    out: np.ndarray,
    output_times: np.ndarray,
    output_directory: Path,
    output_stem: str,
) -> None:
    """Write snapshots, figures and diagnostic time series."""
    x = np.asarray(x, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    out = np.asarray(out, dtype=float)
    output_times = np.asarray(output_times, dtype=float)

    if out.shape[0] != output_times.size:
        raise ValueError(
            f"out contains {out.shape[0]} frames but output_times has "
            f"{output_times.size} entries."
        )
    if initial_state.shape != out.shape[1:]:
        raise ValueError(
            f"initial_state shape {initial_state.shape} is incompatible with "
            f"out shape {out.shape}."
        )
    if output_times.size and (np.any(np.diff(output_times) <= 0.0) or output_times[0] <= 0.0):
        raise ValueError("output_times must be strictly increasing and positive.")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Copies prevent accidental aliasing of a mutable solver work buffer.
    frames = np.concatenate(
        (
            np.array(initial_state, copy=True)[None, :, :],
            np.array(out, copy=True),
        ),
        axis=0,
    )
    times = np.concatenate(([0.0], output_times))
    np.save(output_directory / f"{output_stem}_all_times.npy", times)

    plot_indices = set(_select_plot_indices(frames.shape[0]).tolist())
    dx = model.DOMAIN_LENGTH / x.size
    diagnostic_rows = []

    for frame_index, (time, Q) in enumerate(zip(times, frames)):
        fields = recover_frame(Q)

        if WRITE_ALL_TEXT_FRAMES:
            name = f"{output_stem}_snapshot_{frame_index:04d}_t{time:012.6f}.txt"
            write_snapshot(x, fields, output_directory / name)

        if frame_index in plot_indices:
            name = f"{output_stem}_state_{frame_index:04d}_t{time:012.6f}.png"
            plot_snapshot(x, fields, float(time), output_directory / name)

        finite_residual = np.abs(fields["closure_residual"])
        finite_mismatch = np.abs(fields["traction_mismatch"])
        invalid = np.count_nonzero(fields["closure_status"] != 0.0)
        diagnostic_rows.append(
            [
                time,
                np.nanmin(fields["h_l"]),
                np.nanmax(fields["h_l"]),
                np.nanmin(fields["h_u"]),
                np.nanmax(fields["h_u"]),
                dx * np.nansum(fields["h_l"]),
                dx * np.nansum(fields["h_u"]),
                _harmonic_amplitude(x, fields["h_l"]),
                _harmonic_amplitude(x, fields["eta"]),
                np.nanmin(fields["lambda"]),
                np.nanmax(fields["lambda"]),
                np.nanmin(fields["w_upper"]),
                np.nanmax(fields["tau_b"]),
                invalid,
                np.nanmax(finite_residual) if finite_residual.size else np.nan,
                np.nanmax(finite_mismatch) if finite_mismatch.size else np.nan,
                _hyperbolicity_max_imag(Q),
            ]
        )

    diagnostics = np.asarray(diagnostic_rows, dtype=float)
    header = "\t".join(
        [
            "time",
            "h_lower_min",
            "h_lower_max",
            "h_upper_min",
            "h_upper_max",
            "mass_lower",
            "mass_upper",
            "interface_first_harmonic_amplitude",
            "free_surface_first_harmonic_amplitude",
            "lambda_min",
            "lambda_max",
            "upper_increment_min",
            "tau_b_max",
            "invalid_closure_cells",
            "closure_residual_max",
            "traction_mismatch_max",
            "characteristic_imaginary_part_max_sampled",
        ]
    )
    np.savetxt(
        output_directory / f"{output_stem}_diagnostics.txt",
        diagnostics,
        delimiter="\t",
        header=header,
        comments="",
        fmt="%.12e",
    )

    fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
    plt.plot(diagnostics[:, 0], diagnostics[:, 1], label=r"$\min h_\ell$")
    plt.plot(diagnostics[:, 0], diagnostics[:, 2], label=r"$\max h_\ell$")
    plt.plot(diagnostics[:, 0], diagnostics[:, 3], label=r"$\min h_u$")
    plt.plot(diagnostics[:, 0], diagnostics[:, 4], label=r"$\max h_u$")
    plt.xlabel("dimensionless time")
    plt.ylabel("layer thickness")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    fig.savefig(output_directory / f"{output_stem}_depth_extrema.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
    plt.plot(diagnostics[:, 0], diagnostics[:, 5], label="lower-layer mass")
    plt.plot(diagnostics[:, 0], diagnostics[:, 6], label="upper-layer mass")
    plt.xlabel("dimensionless time")
    plt.ylabel("domain integral")
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.savefig(output_directory / f"{output_stem}_mass_conservation.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
    plt.semilogy(
        diagnostics[:, 0],
        np.maximum(diagnostics[:, 7], 1.0e-18),
        label="interface harmonic",
    )
    plt.semilogy(
        diagnostics[:, 0],
        np.maximum(diagnostics[:, 8], 1.0e-18),
        label="free-surface harmonic",
    )
    plt.xlabel("dimensionless time")
    plt.ylabel("first-harmonic amplitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.savefig(output_directory / f"{output_stem}_harmonic_amplitudes.png", dpi=PLOT_DPI)
    plt.close(fig)

    print(f"Postprocessing complete. Results are in {output_directory.resolve()}")
