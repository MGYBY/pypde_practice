#!/usr/bin/env python3
"""Text and graphic output for the simplified Liu–Mei–Charru solver.

``N_OUTPUTS`` in the main solver is passed to this module as ``n_outputs``.
Every solution frame returned by PyPDE is written as a text snapshot and/or
figure.  The initial condition at ``t = 0`` is retained in the complete NPZ
file and the diagnostic time series, but it is not counted among the
``n_outputs`` PyPDE snapshots.
"""

from pathlib import Path
import numpy as np


DIAGNOSTIC_NAMES = (
    "tau_hat",
    "shields",
    "theta_ratio",
    "eta_ratio",
    "settling_ratio",
    "relaxation",
    "particle_speed",
    "erosion",
    "deposition",
)


def raw_fields(states, velocity_floor, depth_floor):
    """Recover unregularized fields for inspection and output."""
    h, q, up, bed, mobile = states.T
    h0 = np.full_like(h, np.nan)
    valid = np.abs(up) > velocity_floor
    h0[valid] = 3.0 * (h[valid] - q[valid] / up[valid])

    mean_velocity = np.divide(
        q,
        h,
        out=np.full_like(q, np.nan),
        where=np.abs(h) > depth_floor,
    )
    return {
        "h": h,
        "q": q,
        "up": up,
        "bed": bed,
        "mobile": mobile,
        "h0": h0,
        "hp": h - h0,
        "mean_velocity": mean_velocity,
        "free_surface": h + bed,
    }


def closure_fields(states, diagnostic_function):
    """Evaluate all regularized rheology and sediment diagnostics once."""
    values = np.empty((states.shape[0], len(DIAGNOSTIC_NAMES)))
    for i, state in enumerate(states):
        values[i] = diagnostic_function(state)

    fields = {name: values[:, j] for j, name in enumerate(DIAGNOSTIC_NAMES)}
    fields["bedload_flux"] = fields["particle_speed"] * np.maximum(states[:, 4], 0.0)
    return fields


def time_tag(value):
    return f"t_{value:g}".replace("-", "m").replace(".", "p")


def write_parameters(path, parameters):
    with path.open("w", encoding="utf-8") as stream:
        for key, value in parameters.items():
            stream.write(f"{key} = {value}\n")


def finish_figure(fig, path, plt, dpi, show):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_snapshot(x, raw, closure, time_value, path, plt, dpi, show):
    fig, axes = plt.subplots(7, 1, figsize=(9.0, 15.0), sharex=True)

    axes[0].plot(x, raw["h"], label="h")
    axes[0].plot(x, raw["free_surface"], label="b+h")
    axes[0].plot(x, raw["bed"], label="b")
    axes[0].set_ylabel("elevation")
    axes[0].legend(ncol=3)

    axes[1].plot(x, raw["h0"], label="h0")
    axes[1].plot(x, raw["hp"], label="hp")
    axes[1].set_ylabel("layer depth")
    axes[1].legend()

    axes[2].plot(x, raw["q"], label="q")
    axes[2].plot(x, raw["up"], label="u_p")
    axes[2].plot(x, raw["mean_velocity"], label="q/h")
    axes[2].set_ylabel("flow")
    axes[2].legend(ncol=3)

    axes[3].plot(x, raw["mobile"], label="m")
    axes[3].plot(x, closure["bedload_flux"], label="u_s m")
    axes[3].set_ylabel("mobile solid")
    axes[3].legend()

    axes[4].plot(x, closure["shields"], label="Shields")
    axes[4].plot(x, closure["theta_ratio"], label="Theta")
    axes[4].set_ylabel("stress ratio")
    axes[4].legend()

    axes[5].plot(x, closure["eta_ratio"], label="eta_bed/eta_B")
    axes[5].plot(x, closure["settling_ratio"], label="V_s/U_bar")
    axes[5].plot(x, closure["relaxation"], label="T/tau_p")
    axes[5].set_ylabel("settling")
    axes[5].legend()

    axes[6].plot(x, closure["erosion"], label="E")
    axes[6].plot(x, closure["deposition"], label="D")
    axes[6].set_ylabel("exchange")
    axes[6].set_xlabel("x")
    axes[6].legend()

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"Liu-Mei-Charru solution, t = {time_value:.8g}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def snapshot_table(x, raw, closure, packing_fraction):
    fixed_solid = packing_fraction * raw["bed"]
    return np.column_stack(
        (
            x,
            raw["h"], raw["q"], raw["up"], raw["bed"], raw["mobile"],
            raw["h0"], raw["hp"], raw["mean_velocity"], raw["free_surface"],
            closure["tau_hat"], closure["shields"], closure["theta_ratio"],
            closure["eta_ratio"], closure["settling_ratio"], closure["relaxation"],
            closure["particle_speed"], closure["erosion"], closure["deposition"],
            closure["bedload_flux"], fixed_solid, fixed_solid + raw["mobile"],
        )
    )


def postprocess_solution(
    *,
    x,
    Q0,
    out,
    final_time,
    n_outputs,
    output_directory,
    bed_porosity,
    velocity_floor,
    depth_floor,
    diagnostic_function,
    parameters,
    save_snapshot_text=True,
    save_plots=True,
    show_plots=False,
    dpi=220,
):
    """Write all PyPDE output frames, diagnostics, and figures.

    PyPDE returns ``n_outputs`` frames at

        t_i = (i + 1) * final_time / n_outputs,

    so no separate snapshot-time array is needed.
    """
    x = np.asarray(x, dtype=float)
    Q0 = np.asarray(Q0, dtype=float)
    out = np.asarray(out, dtype=float)

    if Q0.ndim != 2 or Q0.shape[1] != 5:
        raise ValueError("Q0 must have shape (nx, 5).")
    if out.ndim != 3 or out.shape[1:] != Q0.shape:
        raise ValueError("out must have shape (n_outputs, nx, 5).")
    if x.shape != (Q0.shape[0],):
        raise ValueError("x must contain one cell-centre coordinate per cell.")
    if n_outputs < 1 or out.shape[0] != n_outputs:
        raise ValueError(
            f"Expected {n_outputs} PyPDE output frames, received {out.shape[0]}."
        )
    if final_time <= 0.0:
        raise ValueError("final_time must be positive.")

    output_times = final_time * np.arange(1, n_outputs + 1, dtype=float) / n_outputs
    all_times = np.concatenate(([0.0], output_times))
    all_states = np.concatenate((Q0[None, :, :], out), axis=0)

    root = Path(output_directory)
    text_dir = root / "text"
    figure_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    if save_snapshot_text:
        text_dir.mkdir(exist_ok=True)
    if save_plots:
        figure_dir.mkdir(exist_ok=True)
        import matplotlib.pyplot as plt
    else:
        plt = None

    np.savez_compressed(
        root / "complete_solution.npz",
        x=x,
        times=all_times,
        states=all_states,
    )
    write_parameters(root / "parameters.txt", parameters)

    packing = 1.0 - bed_porosity
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    diagnostics = []

    snapshot_header = (
        "x h q u_p b m h0 h_p mean_velocity free_surface tau_b_hat "
        "Shields Theta eta_bed_over_eta_B Vs_over_Ubar T_over_tau_p "
        "particle_speed erosion deposition bedload_flux fixed_bed_solid total_local_solid"
    )

    for frame, time_value in enumerate(all_times):
        raw = raw_fields(all_states[frame], velocity_floor, depth_floor)
        closure = closure_fields(all_states[frame], diagnostic_function)

        h_min, h_max = np.nanmin(raw["h"]), np.nanmax(raw["h"])
        b_min, b_max = np.nanmin(raw["bed"]), np.nanmax(raw["bed"])
        total_solid = packing * raw["bed"] + raw["mobile"]

        diagnostics.append(
            [
                time_value,
                dx * np.sum(raw["h"]),
                dx * np.sum(total_solid),
                h_min, h_max, h_max - h_min,
                b_min, b_max, b_max - b_min,
                np.nanmin(raw["h0"]),
                np.nanmin(raw["hp"]),
                np.nanmin(raw["mobile"]),
                np.nanmax(np.abs(closure["tau_hat"])),
                np.nanmax(closure["shields"]),
                np.nanmax(np.abs(closure["bedload_flux"])),
                np.nanmax(np.abs(closure["erosion"] - closure["deposition"])),
                np.nanmax(closure["eta_ratio"]),
                np.nanmin(closure["settling_ratio"]),
                np.nanmax(closure["settling_ratio"]),
                np.nanmin(closure["relaxation"]),
                np.nanmax(closure["relaxation"]),
            ]
        )

        # Frame 0 is the initial condition.  Snapshot files correspond exactly
        # to the n_outputs frames returned by PyPDE.
        if frame == 0:
            continue

        if save_snapshot_text:
            np.savetxt(
                text_dir / f"snapshot_{time_tag(time_value)}.txt",
                snapshot_table(x, raw, closure, packing),
                fmt="%.12e",
                delimiter="\t",
                header=snapshot_header,
            )

        if save_plots:
            plot_snapshot(
                x,
                raw,
                closure,
                time_value,
                figure_dir / f"snapshot_{time_tag(time_value)}.png",
                plt,
                dpi,
                show_plots,
            )

    diagnostics = np.asarray(diagnostics)
    diagnostics[:, 1:3] -= diagnostics[0, 1:3]

    diagnostic_header = (
        "time mud_volume_change total_sediment_change "
        "min_h max_h h_amplitude min_b max_b b_amplitude "
        "min_h0 min_hp min_mobile max_abs_tau_hat max_shields "
        "max_abs_bedload_flux max_abs_exchange_imbalance "
        "max_eta_bed_over_eta_B min_Vs_over_Ubar max_Vs_over_Ubar "
        "min_T_over_tau_p max_T_over_tau_p"
    )
    np.savetxt(
        root / "diagnostics_time_series.txt",
        diagnostics,
        fmt="%.12e",
        delimiter="\t",
        header=diagnostic_header,
    )

    extrema = diagnostics[:, [0, 3, 4, 5, 6, 7, 8]]
    np.savetxt(
        root / "extrema_time_series.txt",
        extrema,
        fmt="%.12e",
        delimiter="\t",
        header="time min_h max_h h_amplitude min_b max_b b_amplitude",
    )

    if save_plots:
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
        axes[0].plot(extrema[:, 0], extrema[:, 1], label="min h")
        axes[0].plot(extrema[:, 0], extrema[:, 2], label="max h")
        axes[0].set_ylabel("fluid depth")
        axes[0].legend()
        axes[1].plot(extrema[:, 0], extrema[:, 4], label="min b")
        axes[1].plot(extrema[:, 0], extrema[:, 5], label="max b")
        axes[1].set_ylabel("bed elevation")
        axes[1].set_xlabel("t")
        axes[1].legend()
        for axis in axes:
            axis.grid(True, alpha=0.25)
        finish_figure(
            fig,
            figure_dir / "extrema_h_b_time_series.png",
            plt,
            dpi,
            show_plots,
        )

        fig, axes = plt.subplots(5, 1, figsize=(8.5, 12.0), sharex=True)
        axes[0].plot(diagnostics[:, 0], diagnostics[:, 1])
        axes[0].set_ylabel("mud-volume change")
        axes[1].plot(diagnostics[:, 0], diagnostics[:, 2])
        axes[1].set_ylabel("solid-volume change")
        axes[2].plot(diagnostics[:, 0], diagnostics[:, 9], label="min h0")
        axes[2].plot(diagnostics[:, 0], diagnostics[:, 10], label="min hp")
        axes[2].set_ylabel("minimum layer")
        axes[2].legend()
        axes[3].plot(diagnostics[:, 0], diagnostics[:, 13], label="max Shields")
        axes[3].plot(diagnostics[:, 0], diagnostics[:, 14], label="max |q_b|")
        axes[3].set_ylabel("morphodynamics")
        axes[3].legend()
        axes[4].plot(diagnostics[:, 0], diagnostics[:, 16], label="max eta ratio")
        axes[4].plot(diagnostics[:, 0], diagnostics[:, 18], label="max V_s/U_bar")
        axes[4].plot(diagnostics[:, 0], diagnostics[:, 20], label="max T/tau_p")
        axes[4].set_ylabel("settling closure")
        axes[4].set_xlabel("t")
        axes[4].legend()
        for axis in axes:
            axis.grid(True, alpha=0.25)
        finish_figure(
            fig,
            figure_dir / "diagnostics_time_series.png",
            plt,
            dpi,
            show_plots,
        )

    print("Postprocessing directory:", root)
    print(f"Saved {n_outputs} PyPDE output snapshots.")
    print("Output times:", [float(value) for value in output_times])
