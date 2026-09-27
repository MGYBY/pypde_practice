#!/usr/bin/env python3
"""Liquid-column projection of the uploaded Basilisk front_runner_ic.h.

Coordinates and time in THIS solver are X=S0*x/H_l and T=S0*t*U_l/H_l.
The positive half-sinusoid occupies lambda/2, not lambda. The compact support
is not periodically tiled and the added volume is not subtracted elsewhere.
"""
from __future__ import annotations
import math
import numpy as np


def localized_scale(x, *, amplitude, wavelength, center):
    """Return s and ds/dX, including an exact constant state outside support.

    s is continuous but its derivative jumps at the two support endpoints,
    matching the uploaded Basilisk initializer (not a smoothed Gaussian).
    Endpoint derivatives are set to zero by convention.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Coordinates must be finite.")
    if not np.isfinite(amplitude) or not 0 <= amplitude < 1:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1.")
    if not np.isfinite(wavelength) or wavelength <= 0 or not np.isfinite(center):
        raise ValueError("wavelength must be positive and center finite.")
    r = (x - center) / wavelength
    active = np.abs(r) < 0.25
    bump = np.where(active, np.cos(2 * np.pi * r), 0.0)
    derivative = np.where(active, -2 * np.pi / wavelength * np.sin(2 * np.pi * r), 0.0)
    return 1.0 + amplitude * bump, amplitude * derivative


def constant_froude_state(x, base_state, *, amplitude, wavelength, center):
    """Set (h_l,h_u)=s*(h_l0,h_u0), (q_l,q_u)=s**1.5*(q_l0,q_u0).

    These are point values at the supplied coordinates, as in the original
    solver. Depth-based Froude numbers are exactly constant in the sampled IC.
    KP stresses/profiles are subsequently obtained from the original closure.
    """
    x = np.asarray(x, dtype=float)
    q0 = np.asarray(base_state, dtype=float)
    if x.ndim != 1 or q0.shape != (4,) or not np.all(np.isfinite(q0)):
        raise ValueError("Expected a finite 1-D coordinate array and base_state shape (4,).")
    if np.any(q0[:2] <= 0):
        raise ValueError("Base depths must be positive.")
    s, _ = localized_scale(x, amplitude=amplitude, wavelength=wavelength, center=center)
    state = np.tile(q0, (x.size, 1))
    state[:, :2] *= s[:, None]
    state[:, 2:] *= s[:, None] ** 1.5
    return np.ascontiguousarray(state)


def exact_excess_volumes(base_state, amplitude, wavelength):
    """Continuum excess layer volumes for a fully contained bump."""
    return np.asarray(base_state, dtype=float)[:2] * amplitude * wavelength / np.pi


def scheduled_times(final_time, interval):
    """Strictly positive output times, always including the exact final time."""
    if not all(np.isfinite(v) and v > 0 for v in (final_time, interval)):
        raise ValueError("final_time and interval must be finite and positive.")
    count = int(math.floor(final_time / interval))
    times = interval * np.arange(1, count + 1, dtype=float)
    tol = 1e-12 * max(1.0, final_time)
    times = times[times < final_time - tol]
    return np.r_[times, float(final_time)]
