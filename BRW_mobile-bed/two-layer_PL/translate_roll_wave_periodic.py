#!/usr/bin/env python3
"""
Periodically translate a steady roll-wave snapshot so that the wave peak
is located at a user-specified x coordinate.

Recommended approach:
    Shift the x coordinates and wrap them periodically, while leaving every
    dependent variable unchanged. Then sort the complete rows by the new x.

This avoids interpolation/smoothing and therefore preserves sharp roll-wave
fronts and discrete/status variables exactly.

Example
-------
python translate_roll_wave_periodic.py \
    two_layer_powerlaw_kp_snapshot_0016_t00016.000000.txt \
    --target-peak 1.0

Optional:
    --peak-column free_surface
    --output translated_peak_x1.txt
    --domain-left 0.0
    --period 5.0
"""

import argparse
from pathlib import Path

import numpy as np


def load_table(filename):
    """Read a whitespace-separated table with one header row."""
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()

    if not header:
        raise ValueError("The input file does not contain a header row.")

    data = np.loadtxt(filename, skiprows=1)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != len(header):
        raise ValueError(
            f"Header has {len(header)} columns but data has {data.shape[1]} columns."
        )

    return header, data


def infer_periodic_domain(x):
    """
    Infer a cell-centered periodic domain from a uniformly spaced x array.

    For the uploaded file, this gives approximately:
        dx = 1/30
        x_left = 0
        period = 5
    """
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        raise ValueError("At least two x points are required.")

    order = np.argsort(x)
    xs = x[order]
    dx_all = np.diff(xs)
    dx = float(np.median(dx_all))

    if dx <= 0:
        raise ValueError("x coordinates must be distinct.")

    # Check that the grid is essentially uniform.
    atol = max(1e-12, 1e-8 * abs(dx))
    if not np.allclose(dx_all, dx, rtol=1e-7, atol=atol):
        raise ValueError(
            "The x grid is not uniformly spaced enough to infer the periodic "
            "domain automatically. Supply --domain-left and --period explicitly."
        )

    # Cell-centered periodic grid:
    # x_0 = x_left + dx/2
    # x_{N-1} = x_left + L - dx/2
    x_left = float(xs[0] - 0.5 * dx)
    period = float(xs[-1] - xs[0] + dx)

    return x_left, period, dx


def wrap_x(x, x_left, period):
    """Wrap coordinates into [x_left, x_left + period)."""
    return x_left + np.mod(x - x_left, period)


def translate_roll_wave(
    input_file,
    output_file,
    target_peak,
    peak_column="free_surface",
    domain_left=None,
    period=None,
):
    header, data = load_table(input_file)

    if "x" not in header:
        raise ValueError("No column named 'x' was found.")

    if peak_column not in header:
        raise ValueError(
            f"Peak column '{peak_column}' was not found.\n"
            f"Available columns: {', '.join(header)}"
        )

    ix = header.index("x")
    ip = header.index(peak_column)

    # Sort the original table by x first.
    original_order = np.argsort(data[:, ix])
    data = data[original_order].copy()

    x = data[:, ix]

    inferred_left, inferred_period, dx = infer_periodic_domain(x)

    if domain_left is None:
        domain_left = inferred_left
    if period is None:
        period = inferred_period

    domain_left = float(domain_left)
    period = float(period)

    if period <= 0:
        raise ValueError("The period must be positive.")

    # Locate the roll-wave peak using the requested variable.
    peak_index = int(np.nanargmax(data[:, ip]))
    old_peak_x = float(data[peak_index, ix])
    peak_value = float(data[peak_index, ip])

    # Normalize the requested coordinate to the periodic domain.
    target_peak_wrapped = float(wrap_x(target_peak, domain_left, period))

    # Use the shortest equivalent periodic translation.
    shift = target_peak_wrapped - old_peak_x
    shift = (shift + 0.5 * period) % period - 0.5 * period

    # IMPORTANT:
    # Only x is changed. All other variables stay in the same row,
    # so every variable is translated by exactly the same amount.
    translated = data.copy()
    translated[:, ix] = wrap_x(
        translated[:, ix] + shift,
        domain_left,
        period,
    )

    # Sort COMPLETE ROWS by the new x coordinate.
    # This keeps all dependent variables consistently attached to each point.
    new_order = np.argsort(translated[:, ix])
    translated = translated[new_order]

    # Numerical cleanup near the periodic boundaries.
    right = domain_left + period
    tol = max(1e-12, 1e-10 * period)
    xx = translated[:, ix]
    xx[np.isclose(xx, right, rtol=0.0, atol=tol)] = domain_left
    translated[:, ix] = xx

    # Re-sort if a near-right-boundary point was mapped to the left boundary.
    translated = translated[np.argsort(translated[:, ix])]

    # Verify where the peak ended up.
    new_peak_index = int(np.nanargmax(translated[:, ip]))
    new_peak_x = float(translated[new_peak_index, ix])

    # Save with the same column order as the input file.
    output_file = Path(output_file)
    np.savetxt(
        output_file,
        translated,
        delimiter="\t",
        header="\t".join(header),
        comments="",
        fmt="%.12e",
    )

    print(f"Input file          : {input_file}")
    print(f"Output file         : {output_file}")
    print(f"Peak variable       : {peak_column}")
    print(f"Peak value          : {peak_value:.12e}")
    print(f"Original peak x     : {old_peak_x:.12e}")
    print(f"Requested peak x    : {target_peak:.12e}")
    print(f"Wrapped target x    : {target_peak_wrapped:.12e}")
    print(f"Periodic shift      : {shift:.12e}")
    print(f"New peak x          : {new_peak_x:.12e}")
    print(f"Domain              : [{domain_left:.12e}, {domain_left + period:.12e})")
    print(f"Period              : {period:.12e}")
    print(f"Inferred grid dx    : {dx:.12e}")

    # The peak should land at the target to roundoff because this method
    # translates coordinates rather than interpolating the profile.
    periodic_error = (
        (new_peak_x - target_peak_wrapped + 0.5 * period) % period
        - 0.5 * period
    )
    print(f"Peak-location error : {periodic_error:.3e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Translate a periodic roll-wave snapshot so that the maximum of a "
            "chosen variable is located at a specified x coordinate."
        )
    )

    parser.add_argument(
        "input_file",
        help="Input whitespace-separated snapshot file.",
    )
    parser.add_argument(
        "--target-peak",
        type=float,
        required=True,
        help="Desired x coordinate of the roll-wave peak.",
    )
    parser.add_argument(
        "--peak-column",
        default="free_surface",
        help="Column used to identify the roll-wave peak (default: free_surface).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename. Default: <input_stem>_translated.txt",
    )
    parser.add_argument(
        "--domain-left",
        type=float,
        default=None,
        help="Left periodic boundary. If omitted, inferred from the x grid.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="Periodic domain length. If omitted, inferred from the x grid.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    if args.output is None:
        output_path = input_path.with_name(
            input_path.stem + "_translated.txt"
        )
    else:
        output_path = Path(args.output)

    translate_roll_wave(
        input_file=input_path,
        output_file=output_path,
        target_peak=args.target_peak,
        peak_column=args.peak_column,
        domain_left=args.domain_left,
        period=args.period,
    )


if __name__ == "__main__":
    main()
