"""Read-only disturbance diagnostics shared with the Basilisk extension.

No flux, source, closure or native solver code is changed by this module.
The DFT acts on equispaced cell-centre heights. Amplitudes are in H_lower.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def height_metrics(x, height, length, mode=1, initial_mean=None):
    x = np.asarray(x, dtype=float)
    h = np.asarray(height, dtype=float)
    if h.ndim != 1 or x.shape != h.shape or not np.isfinite(h).all():
        raise ValueError('Finite, matching one-dimensional coordinates/heights required.')
    if length <= 0 or mode < 1 or int(mode) != mode or 6*mode >= h.size:
        raise ValueError('Require length>0 and at least >6 cells per selected wavelength.')
    mean = float(np.mean(h))
    d = h-mean
    phase = 2*np.pi*int(mode)*x/length
    coefficients = [np.mean(d*np.exp(-1j*j*phase)) for j in (1, 2, 3)]
    c = coefficients[0]
    amplitude = float(2*abs(c))
    rms = float(np.sqrt(np.mean(d*d)))
    return dict(mean=mean, mean_drift=mean-(mean if initial_mean is None else initial_mean),
                rms=rms, linf=float(np.max(np.abs(d))), range=float(np.ptp(h)), min=float(np.min(h)), max=float(np.max(h)),
                creal=float(c.real), cimag=float(c.imag), amplitude=amplitude,
                phase=float(np.angle(c)), harmonic2_amplitude=float(2*abs(coefficients[1])),
                harmonic3_amplitude=float(2*abs(coefficients[2])),
                higher_rms=float(np.sqrt(max(0., rms*rms-amplitude*amplitude/2))))


def growth_row(x, Q, *, time, length, slope, mode=1, initial_means=None, step=0):
    """Q[:,0]=h_lower, Q[:,1]=h_upper; eta_I=h_lower, eta_S=h_lower+h_upper."""
    if initial_means is None:
        initial_means = {}
    row = dict(time=float(time), time_star=float(time/slope), step=int(step),
               ncolumns=len(x), mode=int(mode), k_star=float(2*np.pi*mode*slope/length))
    for name, height in [('surface', Q[:,0]+Q[:,1]), ('interface', Q[:,0])]:
        row.update({name+'_'+k:v for k,v in height_metrics(x,height,length,mode,
                    initial_means.get(name)).items()})
    return row


class GrowthRecorder:
    """CSV writer called at the existing monitor/segment endpoints.

    Sampling does not add native calls or alter integration segmentation. Set
    monitor_interval_slope_scaled for exact regular sample times. Extra output
    endpoints also produce records; always use the recorded time column.
    'step' is the accepted Python segment number, not a native CFL-step count.
    """
    def __init__(self, model, folder, x, Q):
        self.model, self.x = model, np.asarray(x)
        self.mode = int(getattr(model, 'GROWTH_MODE_NUMBER', 1))
        self.enabled = bool(getattr(model, 'GROWTH_ENABLED', False))
        self.initial_means = dict(surface=float(np.mean(Q[:,0]+Q[:,1])),
                                  interface=float(np.mean(Q[:,0])))
        self.stream = None
        self.count = 0
        if not self.enabled:
            return
        groups = dict(Fr_l=model.FR_LOWER, S0=model.SLOPE_TAN, n_l=model.N_LOWER,
                      n_u=model.N_UPPER, rho_r=model.DENSITY_RATIO, h_r=model.DEPTH_RATIO,
                      R_eta_I=model.INTERFACIAL_APPARENT_VISCOSITY_RATIO)
        metadata = dict(solver='corrected Haran Jackson PyPDE; ideal-power-law KP model',
            case_groups=groups,
            geometry=dict(k_star=2*np.pi*self.mode*model.SLOPE_TAN/model.DOMAIN_LENGTH,
                          mode_number=self.mode, domain_length_star=model.DOMAIN_LENGTH/model.SLOPE_TAN),
            normalization=dict(depth='H_l', time='T=S0*t_star', t_star='t*Ubar_l/H_l', amplitude='eta/H_l'),
            coefficient='C=mean((eta-mean(eta))*exp(-i*2*pi*mode*x/L)); A=2*abs(C)',
            quadrature='equal-weight cell-centre DFT; no sinc correction',
            phase='arg(C), radians; unwrap before phase-speed fitting',
            higher_rms='sqrt(max(rms^2-A_mode^2/2,0)); all modes other than tracked fundamental',
            cadence=dict(monitor_interval_T=model.MONITOR_INTERVAL,
                         output_endpoints_also_recorded=True, step='accepted Python segment index; not native CFL steps'),
            initial_means=self.initial_means, front_runner=bool(model.FRONT_RUNNER_ENABLED),
            interpretation=('A localized front runner is broadband: fit individual Fourier modes only before seam interaction; '
                            'global rms and crest height do not have a single eigenvalue.' if model.FRONT_RUNNER_ENABLED
                            else 'Periodic Fourier-mode amplitude is the primary linear growth diagnostic.'))
        if model.BOUNDARY_TYPE != 'periodic':
            metadata['warning']='Nonperiodic domain: Fourier coefficients are descriptive, not periodic temporal eigenmodes.'
        try:
            sigma, vector = model.linear_mode(2*np.pi*self.mode/model.DOMAIN_LENGTH, model.EIGENMODE_BRANCH)
            metadata['shallow_lsa'] = dict(branch=model.EIGENMODE_BRANCH,
                sigma_T=[float(sigma.real), float(sigma.imag)],
                sigma_star=[float(model.SLOPE_TAN*sigma.real), float(model.SLOPE_TAN*sigma.imag)],
                eigenvector_real=vector.real.tolist(), eigenvector_imag=vector.imag.tolist())
        except Exception as exc:
            metadata['shallow_lsa_error']=str(exc)
        folder=Path(folder)
        (folder/'growth_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        row=self.row(0.,Q,0)
        self.stream=(folder/'disturbance_history.csv').open('w',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=list(row))
        self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()

    def row(self,time,Q,step):
        return growth_row(self.x,Q,time=time,length=self.model.DOMAIN_LENGTH,
                slope=self.model.SLOPE_TAN,mode=self.mode,initial_means=self.initial_means,step=step)

    def record(self,time,Q):
        if self.stream is not None:
            self.count+=1
            self.writer.writerow(self.row(time,Q,self.count))
            self.stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream=None
