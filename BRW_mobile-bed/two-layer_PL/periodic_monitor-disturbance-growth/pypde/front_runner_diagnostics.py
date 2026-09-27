"""Low-cost geometry/mass monitoring; expensive constitutive audits at outputs."""
from __future__ import annotations
import math
import numpy as np


def boundary_departures(x, Q, base, *, domain_length, guard_width):
    x, Q, base = np.asarray(x), np.asarray(Q), np.asarray(base)
    left, right = x < guard_width, x > domain_length-guard_width
    if not np.any(left) or not np.any(right):
        raise ValueError('Each guard must contain at least one cell.')
    interface = np.abs(Q[:, 0]-base[0])
    surface = np.abs(Q[:, 0]+Q[:, 1]-base[0]-base[1])
    d = np.maximum(interface, surface)
    return float(np.max(d[left])), float(np.max(d[right]))


def positive_crests(x, y, baseline, threshold):
    """Interior positive local maxima; no claim of persistent wave identity."""
    x, y = np.asarray(x), np.asarray(y)
    ids = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) &
                         (y[1:-1]-baseline > threshold))+1
    positions, heights = x[ids].copy(), y[ids].copy()
    # Subcell parabolic estimate around a single resolved local maximum.
    for j, i in enumerate(ids):
        den = y[i-1]-2*y[i]+y[i+1]
        if den < 0:
            offset = .5*(y[i-1]-y[i+1])/den
            if abs(offset) <= .5:
                positions[j] += offset*(x[i+1]-x[i])
                heights[j] -= .25*(y[i-1]-y[i+1])*offset
    return positions, heights


def cheap_metrics(model, x, Q, base, initial_mass):
    Q = np.asarray(Q)
    if Q.shape != (x.size, 4) or not np.all(np.isfinite(Q)):
        raise FloatingPointError('Non-finite or incorrectly shaped solver state. No clipping/recovery was applied.')
    if np.min(Q[:, :2]) <= model.MIN_DEPTH:
        raise FloatingPointError('A layer depth left the supported wet branch. No depth floor was imposed.')
    dx = model.DOMAIN_LENGTH/x.size
    mass = dx*np.sum(Q[:, :2], axis=0)
    hl, hu = Q[:, 0], Q[:, 1]
    eta = hl+hu
    if model.FRONT_RUNNER_ENABLED:
        left, right = boundary_departures(x, Q, base, domain_length=model.DOMAIN_LENGTH,
                         guard_width=model.FR_GUARD_WAVELENGTHS*model.PERTURBATION_WAVELENGTH)
        threshold = model.FR_TRACKING_HEIGHT
    else:
        left, right, threshold = 0., 0., 1e-8
    px, ph = positive_crests(x, eta, base[0]+base[1], threshold)
    lx, lh = positive_crests(x, hl, base[0], threshold)
    activity = np.maximum(abs(hl-base[0]), abs(eta-base[0]-base[1])) > threshold
    active_ids = np.flatnonzero(activity)
    imax = int(np.argmax(eta))
    return {
        'h_lower_min': float(hl.min()), 'h_lower_max': float(hl.max()),
        'h_upper_min': float(hu.min()), 'h_upper_max': float(hu.max()),
        'surface_min': float(eta.min()), 'surface_max': float(eta.max()),
        'surface_global_peak_x': float(x[imax]),
        'surface_global_peak_excess': float(eta[imax]-base[0]-base[1]),
        'mass_lower': float(mass[0]), 'mass_upper': float(mass[1]),
        'relative_mass_change_lower': float((mass[0]-initial_mass[0])/initial_mass[0]),
        'relative_mass_change_upper': float((mass[1]-initial_mass[1])/initial_mass[1]),
        'guard_left_departure': left, 'guard_right_departure': right,
        'guard_max_departure': max(left, right),
        'surface_crest_count': int(px.size),
        'leading_surface_crest_x': float(px[-1]) if px.size else math.nan,
        'leading_surface_crest_height': float(ph[-1]) if ph.size else math.nan,
        'leading_surface_crest_excess': float(ph[-1]-base[0]-base[1]) if ph.size else math.nan,
        'interface_crest_count': int(lx.size),
        'leading_interface_crest_x': float(lx[-1]) if lx.size else math.nan,
        'leading_interface_crest_height': float(lh[-1]) if lh.size else math.nan,
        'active_region_left_x': float(x[active_ids[0]]) if active_ids.size else math.nan,
        'active_region_right_x': float(x[active_ids[-1]]) if active_ids.size else math.nan,
    }


def sampled_closure_audit(model, Q):
    stride = model.RUNTIME_CELL_STRIDE
    # Always include depth extrema in addition to a deterministic uniform sample.
    ids = np.unique(np.r_[np.arange(0, Q.shape[0], stride), Q.shape[0]-1,
                           np.argmin(Q[:, 0]), np.argmax(Q[:, 0]),
                           np.argmin(Q[:, 1]), np.argmax(Q[:, 1])]).astype(int)
    result = {'audited_cells': int(ids.size), 'sampled_closure_residual_max': 0.,
              'sampled_characteristic_imag_max': 0., 'sampled_upper_increment_min': math.inf}
    for i in ids:
        d = model.closure_diagnostics(Q[i])
        if d['status'] != 0:
            raise model.ClosureError('Output closure audit failed in cell %d: %s' % (i, d))
        result['sampled_closure_residual_max'] = max(result['sampled_closure_residual_max'], abs(d['residual']))
        result['sampled_upper_increment_min'] = min(result['sampled_upper_increment_min'], d['w_upper'])
        eig = model.characteristic_speeds(Q[i], raise_on_complex=False)
        result['sampled_characteristic_imag_max'] = max(result['sampled_characteristic_imag_max'], float(np.max(abs(eig.imag))))
    if result['sampled_characteristic_imag_max'] > model.HYPERBOLICITY_IMAG_TOL:
        raise RuntimeError('Sampled loss of hyperbolicity at an output frame: ' + str(result))
    return result
