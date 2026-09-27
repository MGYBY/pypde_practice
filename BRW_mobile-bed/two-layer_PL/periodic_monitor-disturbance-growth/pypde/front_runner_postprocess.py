"""Stream front-runner frames; distinguish leading crests from global maxima."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def postprocess_run(folder, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    manifest = json.loads((folder/'run_manifest.json').read_text())
    info = json.loads((folder/'initial_summary.json').read_text())
    x = np.load(folder/(model.OUTPUT_STEM+'_x.npy'))
    frames = manifest['frames']
    selected = set(np.unique(np.linspace(0, len(frames)-1, min(model.MAX_PLOT_FRAMES, len(frames))).round().astype(int)))
    figures = folder/'figures'
    figures.mkdir(exist_ok=True)
    texts = folder/'text_frames'
    if model.WRITE_TEXT_FRAMES:
        texts.mkdir(exist_ok=True)
    base = np.asarray(info['base_state'])
    for j, item in enumerate(frames):
        Q = np.load(folder/item['file'], mmap_mode='r')
        if Q.shape != (x.size, 4) or not np.all(np.isfinite(Q)):
            raise ValueError('Invalid saved state: '+item['file'])
        t = item['time']
        if model.WRITE_TEXT_FRAMES:
            table = np.column_stack((x, x/model.SLOPE_TAN, Q[:, :2], Q[:, 0]+Q[:, 1],
                        Q[:, 2:], Q[:, 2]/Q[:, 0], Q[:, 3]/Q[:, 1],
                        model.FR_LOWER*Q[:, 2]/Q[:, 0]**1.5,
                        model.FR_LOWER*Q[:, 3]/Q[:, 1]**1.5))
            np.savetxt(texts/(model.OUTPUT_STEM+'_snapshot_%04d_t%012.6f.txt'%(j,t)),
                table, delimiter='\t', comments='', fmt='%.12e',
                header='X_slope_scaled\tx_basilisk_star\th_lower\th_upper\tfree_surface\tq_lower\tq_upper\tmean_u_lower\tmean_u_upper\tFr_lower\tFr_upper_same_gravity_convention')
        if j in selected:
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            ax.plot(x, Q[:, 0], label='Internal interface')
            ax.plot(x, Q[:, 0]+Q[:, 1], label='Free surface')
            ax.axhline(base[0], linestyle=':', linewidth=.8)
            ax.axhline(base[0]+base[1], linestyle=':', linewidth=.8)
            if model.FRONT_RUNNER_ENABLED:
                active = np.flatnonzero(np.maximum(abs(Q[:, 0]-base[0]),
                                   abs(Q[:, 0]+Q[:, 1]-base[0]-base[1])) > model.FR_TRACKING_HEIGHT)
                if active.size:
                    lam = model.PERTURBATION_WAVELENGTH
                    ax.set_xlim(max(0., x[active[0]]-lam), min(model.DOMAIN_LENGTH, x[active[-1]]+lam))
            ax.set_xlabel(r'$X=S_0x/H_\ell$')
            ax.set_ylabel(r'Height / $H_\ell$')
            title = r'$T=S_0t\overline{U}_\ell/H_\ell=%.6g$'%t
            if item['guard_triggered_or_previously_triggered']:
                title += '  (boundary guard has triggered)'
            ax.set_title(title)
            ax.legend(); ax.grid(True, alpha=.2)
            fig.savefig(figures/('interfaces_%04d.png'%j), dpi=170)
            plt.close(fig)
    np.save(folder/(model.OUTPUT_STEM+'_all_times.npy'), np.asarray([f['time'] for f in frames]))
    history_file = folder/'front_runner_history.tsv'
    if not history_file.exists():
        return
    hist = np.atleast_1d(np.genfromtxt(history_file, delimiter='\t', names=True))
    t = hist['time']
    valid = hist['boundary_guard_ever_triggered'] == 0
    leading = hist['leading_surface_crest_excess'].copy()
    position = hist['leading_surface_crest_x'].copy()
    leading[~valid] = np.nan
    position[~valid] = np.nan
    # Raw finite-difference celerity only when basic crest association tests
    # hold. This is not a persistent crest-ID tracker across births/mergers.
    speed = np.full(t.size, np.nan)
    ambiguous = np.ones(t.size, dtype=int)
    for j in range(1, t.size):
        if valid[j] and valid[j-1] and np.isfinite(position[j:j+1]).all() and np.isfinite(position[j-1]) and hist['surface_crest_count'][j] == hist['surface_crest_count'][j-1]:
            v = (position[j]-position[j-1])/(t[j]-t[j-1])
            # A negative displacement or implausibly large jump is not assigned
            # a physical speed. This threshold is a heuristic, not a proof.
            bound = 3*max(abs(np.asarray(info['base_characteristic_speeds'])))
            if 0 <= v <= bound:
                speed[j], ambiguous[j] = v, 0
    np.savetxt(folder/'leading_crest_candidate.tsv',
        np.column_stack((t, position, leading, speed, ambiguous, ~valid)), delimiter='\t',
        header='time\tleading_surface_crest_x\texcess_height\tcelerity_fd_heuristic\tassociation_ambiguous\tboundary_guard_triggered', comments='', fmt='%.12e')
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    ax.plot(t, hist['surface_global_peak_excess'], label='Global surface peak above base')
    ax.plot(t, leading, label='Downstream-most positive crest candidate (before guard)')
    ax.set_xlabel(r'$T=S_0t\overline{U}_\ell/H_\ell$'); ax.set_ylabel('Height above undisturbed surface')
    ax.legend(); ax.grid(True, alpha=.2)
    fig.savefig(figures/'crest_amplitudes.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    ax.plot(t, position, label='Leading surface-crest candidate')
    ax.plot(t, hist['surface_global_peak_x'], linestyle='--', label='Global maximum location')
    ax.set_xlabel(r'$T$'); ax.set_ylabel(r'$X$'); ax.legend(); ax.grid(True, alpha=.2)
    fig.savefig(figures/'crest_positions.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    ax.plot(t, hist['relative_mass_change_lower'], label='Lower layer')
    ax.plot(t, hist['relative_mass_change_upper'], label='Upper layer')
    ax.set_xlabel(r'$T$'); ax.set_ylabel('Relative integral change from actual initial state')
    ax.legend(); ax.grid(True, alpha=.2)
    fig.savefig(figures/'layer_mass_changes.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    ax.plot(t, hist['guard_left_departure'], label='Left guard')
    ax.plot(t, hist['guard_right_departure'], label='Right guard')
    if model.FRONT_RUNNER_ENABLED:
        ax.axhline(model.FR_GUARD_HEIGHT, linestyle='--', label='Stopping threshold')
    ax.set_xlabel(r'$T$'); ax.set_ylabel('Maximum interface/surface departure from base')
    ax.legend(); ax.grid(True, alpha=.2)
    fig.savefig(figures/'boundary_guard.png', dpi=170); plt.close(fig)
    print('Front-runner text/figures:', folder)
