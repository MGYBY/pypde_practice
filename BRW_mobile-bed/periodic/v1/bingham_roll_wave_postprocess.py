"""Postprocessing for the dimensionless Bingham roll-wave PyPDE model."""

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


ScalarClosure = Callable[[np.ndarray], float]


def raw_fields(
    Q: np.ndarray,
    bed_porosity: float,
    velocity_floor: float,
    depth_floor: float,
    basal_stress_function: ScalarClosure,
    shields_function: ScalarClosure,
    bedload_function: ScalarClosure,
) -> dict:
    """Recover raw layer depths and diagnostic closure values."""
    h, q, up, b = Q.T

    q_over_up = np.divide(
        q,
        up,
        out=np.full_like(q, np.nan),
        where=np.abs(up) > velocity_floor,
    )
    h0 = 3.0 * (h - q_over_up)
    hp = h - h0

    mean_velocity = np.divide(
        q,
        h,
        out=np.full_like(q, np.nan),
        where=np.abs(h) > depth_floor,
    )

    tau_hat = np.array([basal_stress_function(state) for state in Q])
    theta_b = np.array([shields_function(state) for state in Q])
    qb_hat = np.array([bedload_function(state) for state in Q])

    return {
        "h": h,
        "q": q,
        "up": up,
        "b": b,
        "h0": h0,
        "hp": hp,
        "mean_velocity": mean_velocity,
        "free_surface": b + h,
        "tau_hat": tau_hat,
        "theta_b": theta_b,
        "qb_hat": qb_hat,
        "exner_flux": qb_hat / (1.0 - bed_porosity),
    }


def _time_tag(time_value: float) -> str:
    return ("t_%010.4f" % time_value).replace("-", "m").replace(".", "p")


def _selected_indices(times: np.ndarray, requested: Sequence[float]) -> np.ndarray:
    if len(requested) == 0:
        return np.arange(times.size, dtype=int)

    indices = []
    for value in requested:
        index = int(np.argmin(np.abs(times - value)))
        if index not in indices:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def _write_parameters(path: Path, parameters: Mapping[str, object]) -> None:
    lines = [f"{key} = {value}" for key, value in parameters.items()]
    path.write_text("\n".join(lines) + "\n")


def _write_snapshot(path: Path, x: np.ndarray, time_value: float, field: dict) -> None:
    data = np.column_stack(
        (
            x,
            field["h"], field["q"], field["up"], field["b"],
            field["h0"], field["hp"], field["mean_velocity"],
            field["free_surface"], field["tau_hat"], field["theta_b"],
            field["qb_hat"], field["exner_flux"],
        )
    )
    header = (
        f"dimensionless_time = {time_value:.16e}\n"
        "x\th\tq\tu_p\tb\th0_raw\thp_raw\tq_over_h\tb_plus_h\t"
        "tau_b_hat\tTheta_b\tq_b_hat\tq_b_hat_over_1_minus_porosity"
    )
    np.savetxt(path, data, fmt="%.12e", delimiter="\t", header=header)


def _plot_snapshot(path: Path, x: np.ndarray, time_value: float, field: dict,
                   show: bool, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(9.0, 12.0), sharex=True)

    axes[0].plot(x, field["h"], label=r"$h$")
    axes[0].plot(x, field["h0"], label=r"$h_0$")
    axes[0].plot(x, field["hp"], label=r"$h_p$")
    axes[0].set_ylabel("depth")
    axes[0].legend(ncol=3)

    axes[1].plot(x, field["q"], label=r"$q$")
    axes[1].plot(x, field["up"], label=r"$u_p$")
    axes[1].plot(x, field["mean_velocity"], label=r"$q/h$")
    axes[1].set_ylabel("flow")
    axes[1].legend(ncol=3)

    axes[2].plot(x, field["b"], label=r"$b$")
    axes[2].plot(x, field["free_surface"], label=r"$b+h$")
    axes[2].set_ylabel("elevation")
    axes[2].legend(ncol=2)

    axes[3].plot(x, field["tau_hat"], label=r"$\widehat{\tau}_b$")
    axes[3].plot(x, field["theta_b"], label=r"$\Theta_b$")
    axes[3].set_ylabel("stress")
    axes[3].legend(ncol=2)

    axes[4].plot(x, field["qb_hat"], label=r"$\widehat q_b$")
    axes[4].plot(x, field["exner_flux"], label=r"$\widehat q_b/(1-\lambda_p)$")
    axes[4].set_ylabel("bed-load flux")
    axes[4].set_xlabel(r"$x$")
    axes[4].legend(ncol=2)

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(f"Dimensionless time t = {time_value:.6g}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _write_diagnostics(path: Path, x: np.ndarray, times: np.ndarray,
                       states: np.ndarray, field_function) -> np.ndarray:
    dx = x[1] - x[0]
    rows = []

    for time_value, Q in zip(times, states):
        field = field_function(Q)
        h0, hp = field["h0"], field["hp"]
        rows.append([
            time_value,
            np.sum(field["h"]) * dx,
            np.sum(field["q"]) * dx,
            np.mean(field["b"]),
            np.nanmin(h0),
            np.nanmin(hp),
            np.nanmax(np.abs(field["tau_hat"])),
            np.nanmax(field["theta_b"]),
            np.nanmax(np.abs(field["qb_hat"])),
            np.mean((h0 <= 0.0) | (hp <= 0.0) | (field["h"] <= 0.0)),
        ])

    diagnostics = np.asarray(rows)
    header = (
        "time\tint_h_dx\tint_q_dx\tmean_b\tmin_h0_raw\tmin_hp_raw\t"
        "max_abs_tau_b_hat\tmax_Theta_b\tmax_abs_q_b_hat\t"
        "inadmissible_cell_fraction"
    )
    np.savetxt(path, diagnostics, fmt="%.12e", delimiter="\t", header=header)
    return diagnostics


def _plot_diagnostics(path: Path, diagnostics: np.ndarray, show: bool,
                      dpi: int) -> None:
    import matplotlib.pyplot as plt

    time = diagnostics[:, 0]
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 10.0), sharex=True)
    axes[0].plot(time, diagnostics[:, 1])
    axes[0].set_ylabel(r"$\int h\,dx$")
    axes[1].plot(time, diagnostics[:, 3])
    axes[1].set_ylabel(r"mean $b$")
    axes[2].plot(time, diagnostics[:, 4], label=r"min $h_0$")
    axes[2].plot(time, diagnostics[:, 5], label=r"min $h_p$")
    axes[2].set_ylabel("layer minima")
    axes[2].legend(ncol=2)
    axes[3].plot(time, diagnostics[:, 7], label=r"max $\Theta_b$")
    axes[3].plot(time, diagnostics[:, 8], label=r"max $|\widehat q_b|$")
    axes[3].set_ylabel("morphodynamics")
    axes[3].set_xlabel(r"$t$")
    axes[3].legend(ncol=2)

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def postprocess_solution(*, x: np.ndarray, times: np.ndarray, Q0: np.ndarray,
                         out: np.ndarray, output_directory: Path,
                         snapshot_times: Sequence[float], bed_porosity: float,
                         velocity_floor: float, depth_floor: float,
                         basal_stress_function: ScalarClosure,
                         shields_function: ScalarClosure,
                         bedload_function: ScalarClosure,
                         parameters: Mapping[str, object], save_text: bool,
                         save_plots: bool, show_plots: bool, dpi: int) -> None:
    """Write the full archive, selected snapshots, diagnostics, and figures."""
    output_directory.mkdir(parents=True, exist_ok=True)
    text_dir = output_directory / "text"
    figure_dir = output_directory / "figures"
    if save_text:
        text_dir.mkdir(exist_ok=True)
    if save_plots:
        figure_dir.mkdir(exist_ok=True)

    all_times = np.concatenate(([0.0], times))
    all_states = np.concatenate((Q0[None, :, :], out), axis=0)

    np.savez_compressed(
        output_directory / "complete_solution.npz",
        x=x, times=times, Q0=Q0, out=out,
    )
    _write_parameters(output_directory / "parameters.txt", parameters)

    def fields(Q):
        return raw_fields(
            Q, bed_porosity, velocity_floor, depth_floor,
            basal_stress_function, shields_function, bedload_function,
        )

    diagnostics = _write_diagnostics(
        output_directory / "diagnostics_time_series.txt",
        x, all_times, all_states, fields,
    )

    for index in _selected_indices(all_times, snapshot_times):
        time_value = float(all_times[index])
        field = fields(all_states[index])
        tag = _time_tag(time_value)

        if save_text:
            _write_snapshot(text_dir / f"snapshot_{tag}.txt", x, time_value, field)
        if save_plots:
            _plot_snapshot(
                figure_dir / f"snapshot_{tag}.png",
                x, time_value, field, show_plots, dpi,
            )

    if save_plots:
        _plot_diagnostics(
            figure_dir / "diagnostics_time_series.png",
            diagnostics, show_plots, dpi,
        )

    print("Results written to:", output_directory.resolve())
