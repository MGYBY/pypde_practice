"""Exact-time segmented runner with a two-sided periodic-seam guard.

This runner does not change the PDE terms or time discretization. Segmentation is
needed because the public PyPDE API has no Python callback on each CFL step.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import platform
import sys
import traceback
import numpy as np

from front_runner_initialization import (localized_scale, exact_excess_volumes,
                                         scheduled_times)
from front_runner_diagnostics import cheap_metrics, sampled_closure_audit
from growth_diagnostics import GrowthRecorder


def _json_write(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, Path):
            return str(value)
        return value
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def _write_initial(model, folder, x, initial, preflight):
    base = model.uniform_state()
    dx = model.DOMAIN_LENGTH/x.size
    stem = model.OUTPUT_STEM
    np.save(folder/(stem+'_x.npy'), x)
    np.save(folder/(stem+'_Q0.npy'), initial)
    if model._CASE is not None:
        with (folder/'used_case_parameters.ini').open('w', encoding='utf-8') as stream:
            model._CASE.write(stream)
    info = {
        'model': 'two-layer ideal-power-law KP shallow-layer; no capillarity, no air',
        'case_file': model._CASE_PATH,
        'coordinate': 'X = S0*x/H_lower', 'time': 'T = S0*t*Ubar_lower/H_lower',
        'slope_tan': model.SLOPE_TAN,
        'physical_parameters': {key: float(getattr(model, key)) for key in
            ['FR_LOWER', 'DEPTH_RATIO', 'DENSITY_RATIO', 'N_LOWER', 'N_UPPER',
             'INTERFACIAL_APPARENT_VISCOSITY_RATIO', 'SCALED_CONSISTENCY_RATIO']},
        'front_runner_enabled': model.FRONT_RUNNER_ENABLED,
        'ic_mode': model.INITIAL_CONDITION_MODE,
        'domain_length_slope_scaled': model.DOMAIN_LENGTH,
        'domain_length_basilisk_star': model.DOMAIN_LENGTH/model.SLOPE_TAN,
        'requested_final_time_slope_scaled': model.FINAL_TIME,
        'requested_final_time_basilisk_star': model.FINAL_TIME/model.SLOPE_TAN,
        'nx': int(x.size), 'dx_slope_scaled': dx,
        'base_state': base, 'base_source_residual_norm': float(np.linalg.norm(model.S(base))),
        'base_characteristic_speeds': model.characteristic_speeds(base).real,
        'preflight': preflight,
        'sampling': 'cell-centre point values, matching the original PyPDE script',
        'initial_layer_integrals_discrete': dx*np.sum(initial[:, :2], axis=0),
        'python': sys.version, 'platform': platform.platform(),
    }
    if model.FRONT_RUNNER_ENABLED:
        a, lam = model.PERTURBATION_AMPLITUDE, model.PERTURBATION_WAVELENGTH
        xc = model.FR_CENTER_WAVELENGTHS*lam
        s, sx = localized_scale(x, amplitude=a, wavelength=lam, center=xc)
        excess = exact_excess_volumes(base, a, lam)
        info.update({
            'amplitude': a, 'wavelength_slope_scaled': lam,
            'wavelength_basilisk_star': lam/model.SLOPE_TAN,
            'center_slope_scaled': xc,
            'support_slope_scaled': [xc-lam/4, xc+lam/4],
            'cells_per_equivalent_wavelength': lam/dx,
            'cells_per_halfwave_support': lam/(2*dx),
            'excess_layer_volumes_exact': excess,
            'excess_layer_volumes_discrete': dx*np.sum(initial[:, :2]-base[:2], axis=0),
            'mean_layer_depths_exact': base[:2]+excess/model.DOMAIN_LENGTH,
            'stop_before_wrap': model.FR_STOP_BEFORE_WRAP,
            'guard_width_slope_scaled': model.FR_GUARD_WAVELENGTHS*lam,
            'guard_height_threshold': model.FR_GUARD_HEIGHT,
            'monitor_interval_slope_scaled': model.MONITOR_INTERVAL,
        })
    else:
        s, sx = initial[:, 0]/base[0], np.full(x.size, np.nan)
    table = np.column_stack((x, x/model.SLOPE_TAN, s, sx, initial[:, :2],
              initial[:, 0]+initial[:, 1], initial[:, 2:],
              initial[:, 2]/initial[:, 0], initial[:, 3]/initial[:, 1],
              model.FR_LOWER*initial[:, 2]/initial[:, 0]**1.5,
              model.FR_LOWER*initial[:, 3]/initial[:, 1]**1.5))
    np.savetxt(folder/'initial_columns.tsv', table, delimiter='\t', comments='', fmt='%.16e',
         header='X_slope_scaled\tx_basilisk_star\tdepth_factor\tdfactor_dX\th_lower\th_upper\tfree_surface\tq_lower\tq_upper\tUbar_lower\tUbar_upper\tFr_lower\tFr_upper_same_gravity_convention')
    _json_write(folder/'initial_summary.json', info)
    return info


def _save_checkpoint(path, time, Q):
    path = Path(path)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, time=np.asarray(time), Q=Q)
    temp.replace(path)


def run_case(model, *, output_directory=None, dry_run=False,
             force_uncached=False, pde_solver=None):
    """Run a configured model. pde_solver injection is for reproducible tests.

    Outputs are incremental: frame files and a checkpoint survive an early
    guard stop or failure. *_out.npy contains ONLY frames actually saved.
    """
    folder = Path(output_directory if output_directory is not None else model.OUTPUT_DIRECTORY).expanduser().resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Output folder is not empty; choose a new --output folder: '+str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    frame_folder = folder/'frames'
    frame_folder.mkdir()
    stem = model.OUTPUT_STEM
    x = (np.arange(model.NX, dtype=float)+.5)*model.DOMAIN_LENGTH/model.NX
    initial = model.initial_condition(x).copy()
    base = model.uniform_state()
    preflight = model.preflight_initial_state(x, initial)
    if model.FRONT_RUNNER_ENABLED:
        xc = model.FR_CENTER_WAVELENGTHS*model.PERTURBATION_WAVELENGTH
        preflight['analytic_crest'] = model.preflight_initial_state(np.array([xc]), model.initial_condition(np.array([xc])))
    info = _write_initial(model, folder, x, initial, preflight)
    model.print_parameter_summary(preflight)
    print('Coordinates X=S0*x/H_l, time T=S0*t*Ubar_l/H_l')
    if model.FRONT_RUNNER_ENABLED:
        print('Localized support:', info['support_slope_scaled'], '; cells in support:', info['cells_per_halfwave_support'])
        print('Periodic boundary stopping:', model.FR_STOP_BEFORE_WRAP)
        speed = np.max(abs(np.asarray(info['base_characteristic_speeds'])))
        if speed*model.MONITOR_INTERVAL > .5*info['guard_width_slope_scaled']:
            print('WARNING: initial wave speed can traverse > half a guard per check. Reduce monitor_interval_slope_scaled.')
    manifest = {
        'status': 'initialized', 'dry_run': bool(dry_run), 'stop_reason': None,
        'requested_final_time': float(model.FINAL_TIME), 'actual_final_time': 0.,
        'output_directory': str(folder), 'frames': [], 'boundary_guard_first_trigger_time': None,
        'last_checked_guard_clear_time': 0.,
        'time_labels': 'exact native segment endpoints, not nominal intermediate frames',
        'guard_limitation': 'Finite-interval, finite-amplitude geometry detection; not a proof that no subthreshold signal crossed the seam.',
        'native_backend_requested': 'Haran Jackson PyPDE 1.0.0 ADER-WENO',
        'callback_cache_requested': bool(model.CACHE_CALLBACKS and not force_uncached),
        'injected_test_solver': pde_solver is not None,
    }
    np.save(frame_folder/'frame_000000.npy', initial)
    manifest['frames'].append({'index': 0, 'time': 0., 'file': 'frames/frame_000000.npy', 'guard_triggered_or_previously_triggered': False})
    _save_checkpoint(folder/'checkpoint_latest.npz', 0., initial)
    _json_write(folder/'run_manifest.json', manifest)
    if dry_run:
        manifest.update(status='dry_run_complete', stop_reason='dry_run')
        _json_write(folder/'run_manifest.json', manifest)
        print('Initial-state checks complete:', folder)
        return manifest
    try:
        if pde_solver is None:
            from pypde_cached_adapter import get_solver
            pde_solver = get_solver(model.CACHE_CALLBACKS and not force_uncached)
            from pypde_cached_adapter import backend_information
            manifest['native_backend'] = backend_information()
            _json_write(folder/'run_manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='backend_unavailable', stop_reason=str(exc))
        _json_write(folder/'run_manifest.json', manifest)
        raise RuntimeError('The initial state was saved successfully, but the native Haran Jackson PyPDE backend could not be loaded. Install the bundled audited backend using tools/install_native_backend.sh. Details: '+str(exc)) from exc

    base_mass = model.DOMAIN_LENGTH/x.size*np.sum(initial[:, :2], axis=0)
    times = scheduled_times(model.FINAL_TIME, model.OUTPUT_INTERVAL)
    output_index, monitor_index = 0, 1
    current, current_time = initial.copy(), 0.
    first_guard = None
    last_guard_clear = 0.
    saved_files, saved_times = [], []
    monitor_path = folder/'front_runner_history.tsv'
    initial_metrics = cheap_metrics(model, x, current, base, base_mass)
    initial_trigger = bool(model.FRONT_RUNNER_ENABLED and model.BOUNDARY_TYPE == 'periodic' and
                           initial_metrics['guard_max_departure'] > model.FR_GUARD_HEIGHT)
    if initial_trigger:
        first_guard, last_guard_clear = 0., None
        manifest['boundary_guard_first_trigger_time'] = 0.
        manifest['frames'][0]['guard_triggered_or_previously_triggered'] = True
        _json_write(folder/'boundary_guard_event.json', {
            'time': 0., 'previous_checked_time': None,
            'left_departure': initial_metrics['guard_left_departure'],
            'right_departure': initial_metrics['guard_right_departure'],
            'threshold': model.FR_GUARD_HEIGHT,
            'interpretation': 'The INITIAL state already intersects a periodic boundary guard.'})
        _json_write(folder/'run_manifest.json', manifest)
    fieldnames = ['time', 'time_basilisk_star']+list(initial_metrics)+['boundary_guard_ever_triggered']
    audit_rows = []
    failure = None
    growth_recorder = GrowthRecorder(model, folder, x, initial)
    with monitor_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerow(dict(time=0., time_basilisk_star=0., **initial_metrics, boundary_guard_ever_triggered=int(initial_trigger)))
        stream.flush()
        try:
            while current_time < model.FINAL_TIME-1e-12*max(1., model.FINAL_TIME):
                next_monitor = monitor_index*model.MONITOR_INTERVAL
                next_output = float(times[output_index])
                target = min(next_monitor, next_output, model.FINAL_TIME)
                # Protect the preceding, checked state and the true t=0 state
                # against the native routine's in-place mutable work buffer.
                previous, previous_time = current, current_time
                _save_failure_candidate = None
                segment = model._single_pypde_call(pde_solver, previous, target-previous_time, 1)
                if segment.shape != (1, model.NX, 4):
                    raise ValueError('Native solver returned unexpected shape '+str(segment.shape))
                candidate = np.array(segment[-1], copy=True, order='C')
                _save_failure_candidate = candidate
                metric = cheap_metrics(model, x, candidate, base, base_mass)
                # Validate a due frame BEFORE committing its time or history record.
                # A finite returned state can still violate the constitutive branch.
                tol = 1e-11*max(1., target)
                pending_trigger = (model.FRONT_RUNNER_ENABLED and model.BOUNDARY_TYPE == 'periodic' and
                                   metric['guard_max_departure'] > model.FR_GUARD_HEIGHT)
                pending_stopping = bool(pending_trigger and model.FR_STOP_BEFORE_WRAP)
                save_due = next_output <= target+tol or pending_stopping
                if save_due:
                    audit = sampled_closure_audit(model, candidate)
                current, current_time = candidate, float(target)
                print(
                    f"Global T = {current_time:.6f} / {model.FINAL_TIME:g} "
                    f"({100 * current_time / model.FINAL_TIME:.1f}%)",
                    flush=True,
                )
                if next_monitor <= current_time+tol:
                    monitor_index += 1
                trigger = (model.FRONT_RUNNER_ENABLED and model.BOUNDARY_TYPE == 'periodic' and
                           metric['guard_max_departure'] > model.FR_GUARD_HEIGHT)
                if trigger and first_guard is None:
                    first_guard = current_time
                    manifest['boundary_guard_first_trigger_time'] = current_time
                    _save_checkpoint(folder/'last_guard_clear.npz', previous_time, previous)
                    _json_write(folder/'boundary_guard_event.json', {
                        'time': current_time, 'previous_checked_time': previous_time,
                        'left_departure': metric['guard_left_departure'],
                        'right_departure': metric['guard_right_departure'],
                        'threshold': model.FR_GUARD_HEIGHT,
                        'interpretation': 'Geometry departed from the base state near a periodic seam; may be an upstream transient, a wave packet or base-state drift.'})
                if first_guard is None:
                    last_guard_clear = current_time
                writer.writerow(dict(time=current_time, time_basilisk_star=current_time/model.SLOPE_TAN,
                                     **metric, boundary_guard_ever_triggered=int(first_guard is not None)))
                stream.flush()
                growth_recorder.record(current_time, current)
                stopping = bool(trigger and model.FR_STOP_BEFORE_WRAP)
                if save_due:
                    frame_index = len(manifest['frames'])
                    frame_file = frame_folder/('frame_%06d.npy'%frame_index)
                    np.save(frame_file, current)
                    saved_files.append(frame_file)
                    saved_times.append(current_time)
                    manifest['frames'].append({'index': frame_index, 'time': current_time,
                        'file': str(frame_file.relative_to(folder)),
                        'guard_triggered_or_previously_triggered': first_guard is not None})
                    _save_checkpoint(folder/'checkpoint_latest.npz', current_time, current)
                    manifest.update(actual_final_time=current_time,
                                    last_checked_guard_clear_time=last_guard_clear)
                    _json_write(folder/'run_manifest.json', manifest)
                    audit_rows.append(dict(time=current_time, **audit))
                    _json_write(folder/'output_closure_audits.json', audit_rows)
                    print('T=%.8g  max(eta)=%.8g  leading crest X=%.8g  edge departure=%.3e'%(
                        current_time, metric['surface_max'], metric['leading_surface_crest_x'], metric['guard_max_departure']), flush=True)
                if next_output <= current_time+tol:
                    output_index += 1
                if stopping:
                    manifest.update(status='stopped_by_guard', stop_reason='boundary_guard')
                    print('STOP: periodic boundary guard triggered. Triggering frame is flagged, not labelled uncontaminated data.')
                    break
            else:
                manifest.update(status='completed', stop_reason='requested_final_time')
        except Exception as exc:
            failure = exc
            manifest.update(status='failed', stop_reason=str(exc), traceback=traceback.format_exc())
            if 'target' in locals():
                manifest['failed_segment'] = {'start_time': previous_time, 'target_time': float(target),
                    'note': 'This segment was rejected; target_time is not an achieved solution time.'}
            if locals().get('_save_failure_candidate') is not None:
                np.save(folder/'failed_candidate.npy', _save_failure_candidate)
            _save_checkpoint(folder/'last_returned_state.npz', current_time, current)
            if 'previous' in locals():
                _save_checkpoint(folder/'previous_segment_state.npz', previous_time, previous)
        finally:
            growth_recorder.close()
            manifest.update(actual_final_time=current_time, last_checked_guard_clear_time=last_guard_clear)
            _json_write(folder/'run_manifest.json', manifest)
            # Consolidate saved frames without a large in-memory concatenation.
            shape = (len(saved_files), model.NX, 4)
            if saved_files:
                combined = np.lib.format.open_memmap(folder/(stem+'_out.npy'), mode='w+', dtype=float, shape=shape)
                for j, file in enumerate(saved_files):
                    combined[j] = np.load(file, mmap_mode='r')
                combined.flush()
                del combined
            else:
                np.save(folder/(stem+'_out.npy'), np.empty(shape))
            np.save(folder/(stem+'_output_times.npy'), np.asarray(saved_times))
    if failure is not None:
        raise RuntimeError('Run failed; initialized/saved states, history and failure details were retained in '+str(folder)) from failure
    if model.RUN_POSTPROCESSING:
        from front_runner_postprocess import postprocess_run
        try:
            postprocess_run(folder, model)
        except Exception as exc:
            manifest['postprocessing_error'] = str(exc)
            _json_write(folder/'run_manifest.json', manifest)
            raise
    print('Results:', folder)
    return manifest
