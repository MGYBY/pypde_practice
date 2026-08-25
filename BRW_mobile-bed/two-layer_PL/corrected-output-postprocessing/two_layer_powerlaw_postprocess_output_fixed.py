#!/usr/bin/env python3
"""Postprocessing for the two-layer power-law Karman-Pohlhausen model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import two_layer_powerlaw_kp_pypde_output_fixed as model


# Write every saved PyPDE frame.  The initial condition is prepended as frame 0.
WRITE_ALL_TEXT_FRAMES = True

# Plot every saved state by default.  Set this to a positive integer to plot
# only that many approximately evenly spaced saved frames.
NUMBER_OF_PLOTTED_FRAMES = None
TIME_DECIMALS = 9
PLOT_DPI = 220
SHOW_PLOTS = False


def recover_frame(Q: np.ndarray) -> Dict[str, np.ndarray]:
    """Recover velocity-profile and traction diagnostics cell by cell."""
    nx = Q.shape[0]

    h_l = Q[:, 0].copy()
    h_u = Q[:, 1].copy()
    q_l = Q[:, 2].copy()
    q_u = Q[:, 3].copy()

    mean_l = q_l / np.maximum(h_l, model.MIN_DEPTH)
    mean_u = q_u / np.maximum(h_u, model.MIN_DEPTH)

    lam = np.empty(nx)
    tau_b = np.empty(nx)
    tau_i = np.empty(nx)
    u_interface = np.empty(nx)
    w_upper = np.empty(nx)
    momentum_l = np.empty(nx)
    momentum_u = np.empty(nx)
    branch_flag = np.empty(nx)

    for i in range(nx):
        values = model.closure_terms(np.asarray(Q[i], dtype=float))
        (
            lam[i],
            tau_b[i],
            tau_i[i],
            u_interface[i],
            w_upper[i],
            momentum_l[i],
            momentum_u[i],
            branch_flag[i],
        ) = values

    u_surface = u_interface + w_upper

    return {
        "h_l": h_l,
        "h_u": h_u,
        "eta": h_l + h_u,
        "q_l": q_l,
        "q_u": q_u,
        "mean_l": mean_l,
        "mean_u": mean_u,
        "u_interface": u_interface,
        "u_surface": u_surface,
        "w_upper": w_upper,
        "lambda": lam,
        "tau_b": tau_b,
        "tau_i": tau_i,
        "momentum_l": momentum_l,
        "momentum_u": momentum_u,
        "branch_flag": branch_flag,
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
        "tau_I",
        "momentum_lower",
        "momentum_upper",
        "closure_branch_flag",
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
            fields["momentum_l"],
            fields["momentum_u"],
            fields["branch_flag"],
        ]
    )
    return table, "\t".join(names)


def write_snapshot(
    x: np.ndarray,
    fields: Dict[str, np.ndarray],
    filename: Path,
) -> None:
    """Write an Origin-friendly tab-delimited snapshot.

    The first line contains only column names.  Time is encoded in the file
    name and is deliberately not written as a separate metadata header.
    """
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
    """Create a four-panel state summary."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.3), constrained_layout=True)

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
    axes[1, 1].plot(x, fields["branch_flag"], label="closure flag")
    axes[1, 1].set_xlabel(r"$x$")
    axes[1, 1].set_ylabel("closure diagnostic")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"Two-layer power-law model, dimensionless time $t={time:.6g}$")
    fig.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _select_plot_indices(number_of_frames: int) -> np.ndarray:
    if NUMBER_OF_PLOTTED_FRAMES is None:
        return np.arange(number_of_frames, dtype=int)

    count = min(int(NUMBER_OF_PLOTTED_FRAMES), number_of_frames)
    return np.unique(
        np.rint(np.linspace(0, number_of_frames - 1, count)).astype(int)
    )


def postprocess_solution(
    *,
    x: np.ndarray,
    Q0: np.ndarray,
    out: np.ndarray,
    output_times: np.ndarray,
    output_directory: Path,
    output_stem: str,
) -> None:
    """Write snapshots, diagnostic time series, and figures.

    ``output_times`` contains the positive-time states returned by PyPDE.
    The preserved initial condition is prepended explicitly at t=0.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    x = np.asarray(x, dtype=float)
    Q0 = np.array(Q0, dtype=float, copy=True)
    out = np.asarray(out, dtype=float)
    output_times = np.asarray(output_times, dtype=float)

    if Q0.ndim != 2 or Q0.shape[1] != 4:
        raise ValueError("Q0 must have shape (nx, 4).")
    if out.ndim != 3 or out.shape[1:] != Q0.shape:
        raise ValueError("out must have shape (n_outputs, nx, 4).")
    if x.shape != (Q0.shape[0],):
        raise ValueError("x must contain one cell-centre coordinate per cell.")
    if output_times.shape != (out.shape[0],):
        raise ValueError(
            "output_times must contain one time for every PyPDE frame."
        )
    if output_times.size and (
        output_times[0] <= 0.0 or np.any(np.diff(output_times) <= 0.0)
    ):
        raise ValueError("output_times must be strictly increasing and positive.")

    frames = np.concatenate((Q0[np.newaxis, :, :], out), axis=0)
    times = np.concatenate((np.array([0.0]), output_times))

    plot_indices = set(_select_plot_indices(frames.shape[0]).tolist())

    diagnostic_rows = []

    for frame_index, (time, Q) in enumerate(zip(times, frames)):
        fields = recover_frame(Q)

        if WRITE_ALL_TEXT_FRAMES:
            snapshot_name = (
                f"{output_stem}_snapshot_{frame_index:04d}_"
                f"t{time:0{TIME_DECIMALS + 7}.{TIME_DECIMALS}f}.txt"
            )
            write_snapshot(
                x,
                fields,
                output_directory / snapshot_name,
            )

        if frame_index in plot_indices:
            plot_name = (
                f"{output_stem}_state_{frame_index:04d}_"
                f"t{time:0{TIME_DECIMALS + 7}.{TIME_DECIMALS}f}.png"
            )
            plot_snapshot(
                x,
                fields,
                float(time),
                output_directory / plot_name,
            )

        dx = model.DOMAIN_LENGTH / x.size
        diagnostic_rows.append(
            [
                time,
                np.min(fields["h_l"]),
                np.max(fields["h_l"]),
                np.min(fields["h_u"]),
                np.max(fields["h_u"]),
                dx * np.sum(fields["h_l"]),
                dx * np.sum(fields["h_u"]),
                np.min(fields["lambda"]),
                np.max(fields["lambda"]),
                np.min(fields["w_upper"]),
                np.max(fields["tau_b"]),
                np.count_nonzero(fields["branch_flag"] < 0.5),
            ]
        )

    diagnostics = np.asarray(diagnostic_rows, dtype=float)
    diagnostic_header = "\t".join(
        [
            "time",
            "h_lower_min",
            "h_lower_max",
            "h_upper_min",
            "h_upper_max",
            "mass_lower",
            "mass_upper",
            "lambda_min",
            "lambda_max",
            "upper_increment_min",
            "tau_b_max",
            "invalid_closure_cells",
        ]
    )
    np.savetxt(
        output_directory / f"{output_stem}_diagnostics.txt",
        diagnostics,
        delimiter="\t",
        header=diagnostic_header,
        comments="",
        fmt="%.12e",
    )

    # Separate time-series figures keep conservation and amplitude checks clear.
    fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
    plt.plot(diagnostics[:, 0], diagnostics[:, 1], label=r"$\min h_\ell$")
    plt.plot(diagnostics[:, 0], diagnostics[:, 2], label=r"$\max h_\ell$")
    plt.plot(diagnostics[:, 0], diagnostics[:, 3], label=r"$\min h_u$")
    plt.plot(diagnostics[:, 0], diagnostics[:, 4], label=r"$\max h_u$")
    plt.xlabel("dimensionless time")
    plt.ylabel("layer thickness")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    fig.savefig(
        output_directory / f"{output_stem}_depth_extrema.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
    plt.plot(diagnostics[:, 0], diagnostics[:, 5], label="lower-layer mass")
    plt.plot(diagnostics[:, 0], diagnostics[:, 6], label="upper-layer mass")
    plt.xlabel("dimensionless time")
    plt.ylabel("domain integral")
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.savefig(
        output_directory / f"{output_stem}_mass_conservation.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "Postprocessing complete.  Results are in "
        f"{output_directory.resolve()}"
    )
