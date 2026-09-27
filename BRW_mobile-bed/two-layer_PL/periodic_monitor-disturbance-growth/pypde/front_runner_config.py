"""Strict, single-case-per-process INI loader for the original PyPDE model."""
from __future__ import annotations
import configparser
import math
import os
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]
# External keys -> (original/new module global, conversion, default).
SCHEMA = {
    'disturbance_diagnostics': {
        'enabled': ('GROWTH_ENABLED', bool, False),
        'mode_number': ('GROWTH_MODE_NUMBER', int, 1),
    },
    'physical_problem': {
        'froude_lower': ('FR_LOWER', float, .6),
        'slope_tan': ('SLOPE_TAN', float, .06),
        'depth_ratio': ('DEPTH_RATIO', float, 1.),
        'density_ratio': ('DENSITY_RATIO', float, .8),
        'interfacial_apparent_viscosity_ratio': ('INTERFACIAL_APPARENT_VISCOSITY_RATIO', float, 1.544),
    },
    'lower_rheology': {'power_index': ('N_LOWER', float, .4)},
    'upper_rheology': {'power_index': ('N_UPPER', float, .8)},
    'domain': {
        'length_slope_scaled': ('DOMAIN_LENGTH', float, 151.1),
        'nx': ('NX', int, 12000),
        'boundary_type': ('BOUNDARY_TYPE', str, 'periodic'),
    },
    'time_integration': {
        'end_time_slope_scaled': ('FINAL_TIME', float, 90.),
        'output_interval_slope_scaled': ('OUTPUT_INTERVAL', float, 1.),
        'monitor_interval_slope_scaled': ('MONITOR_INTERVAL', float, .05),
        'cfl': ('CFL', float, .5),
        'reconstruction_order': ('RECONSTRUCTION_ORDER', int, 2),
        'numerical_flux': ('NUMERICAL_FLUX', str, 'rusanov'),
        'stiff_source': ('STIFF_SOURCE', bool, False),
        'n_threads': ('N_THREADS', int, 1),
        'cache_callbacks': ('CACHE_CALLBACKS', bool, True),
    },
    'front_runner': {
        'enabled': ('FRONT_RUNNER_ENABLED', bool, True),
        'amplitude': ('FR_AMPLITUDE', float, .1),
        'wavelength_slope_scaled': ('FR_WAVELENGTH', float, 2.),
        'center_wavelengths': ('FR_CENTER_WAVELENGTHS', float, .75),
        'stop_before_wrap': ('FR_STOP_BEFORE_WRAP', bool, True),
        'boundary_guard_wavelengths': ('FR_GUARD_WAVELENGTHS', float, .25),
        'boundary_guard_height': ('FR_GUARD_HEIGHT', float, .01),
        'tracking_height': ('FR_TRACKING_HEIGHT', float, .001),
    },
    'periodic': {
        'initial_condition_mode': ('PERIODIC_IC_MODE', str, 'linear_eigenmode'),
        'amplitude': ('PERIODIC_AMPLITUDE', float, .001),
        'wavelength_slope_scaled': ('PERIODIC_WAVELENGTH', str, 'domain'),
        'eigenmode_branch': ('EIGENMODE_BRANCH', str, 'free_surface'),
        'upper_layer_phase_shift': ('UPPER_LAYER_PHASE_SHIFT', float, 0.),
    },
    'diagnostics': {
        'preflight_every_cell': ('PREFLIGHT_EVERY_CELL', bool, True),
        'runtime_cell_stride': ('RUNTIME_CELL_STRIDE', int, 64),
        'minimum_cells_per_support': ('MIN_CELLS_PER_SUPPORT', int, 16),
    },
    'output': {
        'directory': ('OUTPUT_DIRECTORY', Path, 'front_runner_output'),
        'stem': ('OUTPUT_STEM', str, 'two_layer_powerlaw_front_runner'),
        'postprocess': ('RUN_POSTPROCESSING', bool, True),
        'write_text_frames': ('WRITE_TEXT_FRAMES', bool, True),
        'max_plot_frames': ('MAX_PLOT_FRAMES', int, 12),
    },
}


def load_case_settings(path=None):
    raw = str(path if path is not None else os.environ.get('TLPL_CASE', ROOT/'case_parameters.ini'))
    if raw == 'legacy':
        # Used only by the included original-code regression suite. Keep the
        # uploaded Python defaults and its periodic eigenmode exactly intact.
        return None, None, {'FRONT_RUNNER_ENABLED': False}
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Case INI does not exist: ' + str(path))
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
    with path.open(encoding='utf-8') as stream:
        cp.read_file(stream)
    if cp.defaults():
        raise ValueError('Use explicit sections, not a [DEFAULT] section.')
    for section in cp.sections():
        if section not in SCHEMA:
            raise ValueError('Unknown case section [' + section + ']. Use import_basilisk_case.py for a Basilisk INI.')
        extra = set(cp[section]) - set(SCHEMA[section])
        if extra:
            raise ValueError('Unknown option(s) in [' + section + ']: ' + ', '.join(sorted(extra)))
    values = {}
    for section, fields in SCHEMA.items():
        for key, (global_name, cast, default) in fields.items():
            if not cp.has_section(section):
                cp.add_section(section)
            if not cp.has_option(section, key):
                cp.set(section, key, str(default).lower() if cast is bool else str(default))
            if cast is bool:
                value = cp.getboolean(section, key)
            else:
                value = cast(cp.get(section, key))
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(section + '.' + key + ' must be finite.')
            values[global_name] = value
    for key in ['SLOPE_TAN', 'MONITOR_INTERVAL', 'RUNTIME_CELL_STRIDE', 'MAX_PLOT_FRAMES', 'MIN_CELLS_PER_SUPPORT']:
        if values[key] <= 0:
            raise ValueError(key + ' must be positive.')
    if Path(values['OUTPUT_STEM']).name != values['OUTPUT_STEM'] or not values['OUTPUT_STEM']:
        raise ValueError('output.stem must be a non-empty filename stem, not a path.')
    if values['N_THREADS'] > values['NX']:
        raise ValueError('time_integration.n_threads must not exceed domain.nx.')
    if not 1 <= values['RECONSTRUCTION_ORDER'] <= values['NX']:
        raise ValueError('Reconstruction order must be between 1 and domain.nx.')
    if values['N_THREADS'] < 1:
        from multiprocessing import cpu_count
        values['N_THREADS'] = min(values['NX'], max(1, cpu_count()-1))
        cp.set('time_integration', 'n_threads', str(values['N_THREADS']))
    values['NUMERICAL_FLUX'] = values['NUMERICAL_FLUX'].lower()
    cp.set('time_integration', 'numerical_flux', values['NUMERICAL_FLUX'])
    values['OUTPUT_MODE'] = 'exact_segmented'
    values['PERTURBATION_ENVELOPE'] = 'periodic'  # legacy modes only
    if values['FRONT_RUNNER_ENABLED']:
        values['INITIAL_CONDITION_MODE'] = 'front_runner_constant_froude'
        values['PERTURBATION_AMPLITUDE'] = values['FR_AMPLITUDE']
        values['PERTURBATION_WAVELENGTH'] = values['FR_WAVELENGTH']
        validate_front_runner(values)
    else:
        values['INITIAL_CONDITION_MODE'] = values['PERIODIC_IC_MODE']
        if values['INITIAL_CONDITION_MODE'] == 'front_runner_constant_froude':
            raise ValueError('Use front_runner.enabled=true for a localized disturbance.')
        values['PERTURBATION_AMPLITUDE'] = values['PERIODIC_AMPLITUDE']
        lam = values['PERIODIC_WAVELENGTH']
        values['PERTURBATION_WAVELENGTH'] = values['DOMAIN_LENGTH'] if lam == 'domain' else float(lam)
    return cp, path, values


def validate_front_runner(v):
    a, lam, length = v['FR_AMPLITUDE'], v['FR_WAVELENGTH'], v['DOMAIN_LENGTH']
    if not 0 <= a < 1 or lam <= 0 or length <= 0:
        raise ValueError('Require 0 <= amplitude < 1, positive wavelength and domain length.')
    xc = v['FR_CENTER_WAVELENGTHS'] * lam
    left, right = xc-lam/4, xc+lam/4
    if not 0 < left < right < length:
        raise ValueError('The ENTIRE localized support must lie strictly inside the domain.')
    if v['FR_GUARD_WAVELENGTHS'] <= 0 or v['FR_GUARD_HEIGHT'] <= 0 or v['FR_TRACKING_HEIGHT'] <= 0:
        raise ValueError('Guard width, guard height, and tracking height must be positive.')
    width = v['FR_GUARD_WAVELENGTHS'] * lam
    if 2*width >= length:
        raise ValueError('Boundary guards must not overlap.')
    dx = length/v['NX'] if v['NX'] > 0 else math.inf
    if width < dx:
        raise ValueError('Boundary guard must span at least one complete cell.')
    if v['BOUNDARY_TYPE'] not in {'periodic', 'transitive'}:
        raise ValueError('boundary_type must be periodic or transitive.')
    if v['FR_STOP_BEFORE_WRAP'] and v['BOUNDARY_TYPE'] != 'periodic':
        raise ValueError('stop_before_wrap requires periodic boundaries; disable it for transitive boundaries.')
    if v['FR_STOP_BEFORE_WRAP'] and not (left > width and right < length-width):
        raise ValueError('Initial support overlaps a boundary guard. Increase domain/center or reduce guard width.')
    if lam/(2*dx) < v['MIN_CELLS_PER_SUPPORT']:
        raise ValueError('Too few cells in the half-wave support. Increase domain.nx or reduce diagnostics.minimum_cells_per_support explicitly.')
    if lam/(2*dx) < 50:
        warnings.warn('Fewer than 50 cells resolve the localized half-wave; use this as a smoke test, not a converged result.')
