#!/usr/bin/env python3
"""Postprocessing for ``newtonian_charru_pypde.py``.

The module writes the complete solution, selected text snapshots, diagnostic
history, and PNG figures.  All quantities are dimensionless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


ScalarFunction = Callable[[np.ndarray], float]
CharruFunction = Callable[[np.ndarray], tuple]


def _safe_time_tag(time_value: float) -> str:
    return (f"t_{time_value:g}").replace("-", "m").replace(".", "p")


def _selected_indices(all_times: np.ndarray, requested_times: Sequence[float]) -> np.ndarray:
    requested = np.asarray(requested_times, dtype=float)
    if requested.size == 0:
        return np.arange(all_times.size, dtype=int)

    selected: list[int] = []
    for requested_time in requested:
        index = int(np.argmin(np.abs(all_times - requested_time)))
        if index not in selected:
            selected.append(index)
    return np.asarray(selected, dtype=int)


def _write_parameters(path: Path, parameters: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for key, value in parameters.items():
            stream.write(f"{key} = {value}\n")


def _raw_fields(states: np.ndarray, depth_floor: float) -> dict[str, np.ndarray]:
    h = states[:, 0]
    q = states[:, 1]
    bed = states[:, 2]
    mobile = states[:, 3]
    mean_velocity = np.divide(
        q,
        h,
        out=np.full_like(q, np.nan),
        where=np.abs(h) > depth_floor,
    )
    return {
        "h": h,
        "q": q,
        "bed": bed,
        "mobile": mobile,
        "mean_velocity": mean_velocity,
        "free_surface": h + bed,
    }


def _closure_fields(
    states: np.ndarray,
    basal_stress_function: ScalarFunction,
    charru_terms_function: CharruFunction,
) -> dict[str, np.ndarray]:
    number_of_cells = states.shape[0]
    stress_ratio = np.empty(number_of_cells)
    particle_speed = np.empty(number_of_cells)
    erosion = np.empty(number_of_cells)
    deposition = np.empty(number_of_cells)
    shields = np.empty(number_of_cells)
    theta_ratio = np.empty(number_of_cells)
    mobile_equilibrium = np.empty(number_of_cells)

    for i in range(number_of_cells):
        state = states[i]
        stress_ratio[i] = basal_stress_function(state)
        (
            particle_speed[i],
            erosion[i],
            deposition[i],
            shields[i],
            theta_ratio[i],
            mobile_equilibrium[i],
        ) = charru_terms_function(state)

    mobile_for_flux = np.maximum(states[:, 3], 0.0)
    mobile_flux = particle_speed * mobile_for_flux

    return {
        "stress_ratio": stress_ratio,
        "particle_speed": particle_speed,
        "erosion": erosion,
        "deposition": deposition,
        "shields": shields,
        "theta_ratio": theta_ratio,
        "mobile_equilibrium": mobile_equilibrium,
        "mobile_flux": mobile_flux,
    }


def postprocess_solution(
    *,
    x: np.ndarray,
    times: np.ndarray,
    Q0: np.ndarray,
    out: np.ndarray,
    output_directory: Path,
    snapshot_times: Sequence[float],
    bed_porosity: float,
    depth_floor: float,
    basal_stress_function: ScalarFunction,
    charru_terms_function: CharruFunction,
    parameters: Mapping[str, object],
    save_text: bool = True,
    save_plots: bool = True,
    show_plots: bool = False,
    dpi: int = 220,
) -> None:
    """Save the complete history and selected diagnostic snapshots."""
    x = np.asarray(x, dtype=float)
    times = np.asarray(times, dtype=float)
    Q0 = np.asarray(Q0, dtype=float)
    out = np.asarray(out, dtype=float)

    if Q0.ndim != 2 or Q0.shape[1] != 4:
        raise ValueError("Q0 must have shape (nx, 4).")
    if out.ndim != 3 or out.shape[1:] != Q0.shape:
        raise ValueError("out must have shape (nt, nx, 4).")
    if times.shape != (out.shape[0],):
        raise ValueError("times must contain one value per returned PyPDE frame.")
    if x.shape != (Q0.shape[0],):
        raise ValueError("x must contain one cell-centre coordinate per cell.")

    output_directory = Path(output_directory)
    text_directory = output_directory / "text"
    figure_directory = output_directory / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    if save_text:
        text_directory.mkdir(parents=True, exist_ok=True)
    if save_plots:
        figure_directory.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
    else:
        plt = None

    all_times = np.concatenate(([0.0], times))
    all_states = np.concatenate((Q0[None, :, :], out), axis=0)
    selected = _selected_indices(all_times, snapshot_times)

    # np.savez_compressed(
    #     output_directory / "complete_solution.npz",
    #     x=x,
    #     times=all_times,
    #     states=all_states,
    # )
    _write_parameters(output_directory / "parameters.txt", parameters)

    packing_fraction = 1.0 - bed_porosity
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    diagnostic_rows = []

    for frame, time_value in enumerate(all_times):
        raw = _raw_fields(all_states[frame], depth_floor)
        closure = _closure_fields(
            all_states[frame],
            basal_stress_function,
            charru_terms_function,
        )

        mud_volume = dx * np.sum(raw["h"])
        total_solid = dx * np.sum(packing_fraction * raw["bed"] + raw["mobile"])

        # diagnostic_rows.append(
        #     [
        #         time_value,
        #         mud_volume,
        #         total_solid,
        #         np.nanmin(raw["h"]),
        #         np.nanmin(raw["mobile"]),
        #         np.nanmax(np.abs(closure["stress_ratio"])),
        #         np.nanmax(closure["shields"]),
        #         np.nanmax(np.abs(closure["mobile_flux"])),
        #         np.nanmax(np.abs(closure["erosion"] - closure["deposition"])),
        #     ]
        # )

        diagnostic_rows.append(
            [
                time_value,
                np.nanmin(raw["h"]),
                np.nanmax(raw["h"]),
                np.nanmin(raw["bed"]),
                np.nanmax(raw["bed"]),
                np.nanmin(raw["mobile"]),
                np.nanmax(raw["mobile"]),
                np.nanmax(np.abs(closure["erosion"] - closure["deposition"])),
            ]
        )

    diagnostics = np.asarray(diagnostic_rows, dtype=float)
    diagnostics[:, 1] -= diagnostics[0, 1]
    diagnostics[:, 2] -= diagnostics[0, 2]

    np.savetxt(
        output_directory / "diagnostics_time_series.txt",
        diagnostics,
        fmt="%.12e",
        delimiter="\t",
        # header=(
        #     "time mud_volume_change total_solid_change min_h min_m "
        #     "max_abs_tau_ratio max_shields max_abs_mobile_flux "
        #     "max_abs_exchange_imbalance"
        # ),
    )

    column_header = (
        "x h q b m mean_velocity free_surface tau_ratio Shields Theta "
        "particle_speed m_equilibrium erosion deposition mobile_flux "
        "packed_bed_solid total_local_solid"
    )

    for index in selected:
        time_value = float(all_times[index])
        raw = _raw_fields(all_states[index], depth_floor)
        closure = _closure_fields(
            all_states[index],
            basal_stress_function,
            charru_terms_function,
        )

        packed_bed_solid = packing_fraction * raw["bed"]
        total_local_solid = packed_bed_solid + raw["mobile"]

        if save_text:
            table = np.column_stack(
                (
                    x,
                    raw["h"],
                    raw["q"],
                    raw["bed"],
                    raw["mobile"],
                    raw["mean_velocity"],
                    raw["free_surface"],
                    closure["stress_ratio"],
                    closure["shields"],
                    closure["theta_ratio"],
                    closure["particle_speed"],
                    closure["mobile_equilibrium"],
                    closure["erosion"],
                    closure["deposition"],
                    closure["mobile_flux"],
                    packed_bed_solid,
                    total_local_solid,
                )
            )
            header = f"dimensionless_time = {time_value:.16e}\n{column_header}"
            np.savetxt(
                text_directory / f"snapshot_{_safe_time_tag(time_value)}.txt",
                table,
                fmt="%.12e",
                delimiter="\t",
                header=header,
            )

        if save_plots:
            fig, axes = plt.subplots(5, 1, figsize=(9.0, 11.5), sharex=True)

            axes[0].plot(x, raw["h"], label="h")
            axes[0].plot(x, raw["free_surface"], label="b+h")
            axes[0].plot(x, raw["bed"], label="b")
            axes[0].set_ylabel("elevation")
            axes[0].legend(loc="best", ncol=3)

            axes[1].plot(x, raw["q"], label="q")
            axes[1].plot(x, raw["mean_velocity"], label="q/h")
            axes[1].set_ylabel("flow")
            axes[1].legend(loc="best")

            axes[2].plot(x, closure["stress_ratio"], label="tau_b/tau_b0")
            axes[2].plot(x, closure["shields"], label="Shields")
            axes[2].plot(x, closure["theta_ratio"], label="Theta")
            axes[2].set_ylabel("stress")
            axes[2].legend(loc="best", ncol=3)

            axes[3].plot(x, raw["mobile"], label="m")
            axes[3].plot(x, closure["mobile_equilibrium"], label="m_eq")
            axes[3].plot(x, closure["mobile_flux"], label="u_s m")
            axes[3].set_ylabel("mobile solid")
            axes[3].legend(loc="best", ncol=3)

            axes[4].plot(x, closure["erosion"], label="E")
            axes[4].plot(x, closure["deposition"], label="D")
            axes[4].set_ylabel("exchange")
            axes[4].set_xlabel("x")
            axes[4].legend(loc="best")

            for axis in axes:
                axis.grid(True, alpha=0.25)

            fig.suptitle(f"Newtonian-Charru solution, t = {time_value:.8g}")
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
            fig.savefig(
                figure_directory / f"snapshot_{_safe_time_tag(time_value)}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    # if save_plots:
    #     fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.5), sharex=True)
    #     axes[0].plot(diagnostics[:, 0], diagnostics[:, 1])
    #     axes[0].set_ylabel("mud-volume change")
    #     axes[1].plot(diagnostics[:, 0], diagnostics[:, 2])
    #     axes[1].set_ylabel("solid-volume change")
    #     axes[2].plot(diagnostics[:, 0], diagnostics[:, 3], label="min h")
    #     axes[2].plot(diagnostics[:, 0], diagnostics[:, 4], label="min m")
    #     axes[2].set_ylabel("minimum state")
    #     axes[2].legend(loc="best")
    #     axes[3].plot(diagnostics[:, 0], diagnostics[:, 6], label="max Shields")
    #     axes[3].plot(diagnostics[:, 0], diagnostics[:, 7], label="max |u_s m|")
    #     axes[3].set_ylabel("morphodynamics")
    #     axes[3].set_xlabel("t")
    #     axes[3].legend(loc="best")
    #     for axis in axes:
    #         axis.grid(True, alpha=0.25)
    #     fig.tight_layout()
    #     fig.savefig(
    #         figure_directory / "diagnostics_time_series.png",
    #         dpi=dpi,
    #         bbox_inches="tight",
    #     )
    #     if show_plots:
    #         plt.show()
    #     else:
    #         plt.close(fig)

    print("Postprocessing directory:", output_directory)
    print("Saved snapshot times:", [float(all_times[i]) for i in selected])
