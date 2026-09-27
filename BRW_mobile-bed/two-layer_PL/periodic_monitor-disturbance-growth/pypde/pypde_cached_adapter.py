"""Reuse PyPDE 1.0.0's compiled callbacks across exact-time segments.

This does NOT implement a different numerical method. It calls the same native
pde_solver symbol, using the same ctypes signature and arguments as upstream
pypde.solvers.pde_solver. It avoids repeated Numba compilation only. Private
upstream utility imports are isolated here; --uncached uses the public API.

Source audited: https://github.com/haranjackson/PyPDE/blob/master/pypde/solvers.py
and pypde/{cfuncs,utils}.py, accessed 2026-09-23.
"""
from __future__ import annotations
from multiprocessing import cpu_count
import numpy as np


class CachedPyPDESolver:
    def __init__(self):
        try:
            from pypde.cfuncs import generate_cfuncs
            from pypde.utils import create_solver, c_ptr, parse_boundary_types
        except (ImportError, AttributeError) as exc:
            raise ImportError('Requires Haran Jackson PyPDE 1.0.0 (import pypde), not py-pde (import pde).') from exc
        self.generate_cfuncs = generate_cfuncs
        self.native = create_solver()
        self.c_ptr = c_ptr
        self.parse_boundary_types = parse_boundary_types
        self.functions = None
        self.key = None
        self.backend_info = backend_information()
        require_audited_backend(self.backend_info)

    def __call__(self, Q0, tf, L, *, F, B, S, boundaryTypes='periodic', cfl=.5,
                 order=2, ndt=1, flux='rusanov', stiff=False, nThreads=1):
        Q0 = np.asarray(Q0)
        if Q0.ndim != 2 or Q0.shape[1] != 4 or Q0.dtype != np.float64 or not Q0.flags.c_contiguous:
            raise ValueError('Adapter expects a contiguous float64 state (NX, 4).')
        if (len(L) != 1 or not np.isfinite(L[0]) or L[0] <= 0 or
                tf <= 0 or not np.isfinite(tf) or int(ndt) != ndt or ndt < 1):
            raise ValueError('Invalid one-dimensional integration segment.')
        if not np.all(np.isfinite(Q0)):
            raise ValueError('Native input state must be finite.')
        if not Q0.flags.writeable:
            raise ValueError('Native input state must be a writable private work array.')
        if not 0 < cfl < 1 or not np.isfinite(cfl):
            raise ValueError('CFL must be finite and between zero and one.')
        if int(order) != order or not 1 <= order <= Q0.shape[0]:
            raise ValueError('Reconstruction order must be an integer between 1 and NX.')
        if flux not in {'rusanov', 'roe', 'osher'}:
            raise ValueError('Unknown native flux: ' + str(flux))
        if not np.isfinite(nThreads) or int(nThreads) != nThreads:
            raise ValueError('Thread count must be an integer.')
        if nThreads > Q0.shape[0]:
            raise ValueError('Thread count must not exceed NX.')
        key = (F, B, S, 1, 4)
        if self.key != key:
            # Retain cfunc objects for the lifetime of every native invocation.
            self.functions = self.generate_cfuncs(F, B, S, 1, 4)
            self.key = key
        cf, cb, cs = self.functions
        nx = np.asarray([Q0.shape[0]], dtype=np.int32)
        dx = np.asarray([float(L[0])/Q0.shape[0]], dtype=np.float64)
        boundaries = self.parse_boundary_types(boundaryTypes, 1)
        ret = np.zeros((int(ndt),) + Q0.shape, dtype=np.float64)
        flux_code = {'rusanov': 0, 'roe': 1, 'osher': 2}[flux]
        threads = int(nThreads) if nThreads >= 1 else min(Q0.shape[0], max(1, cpu_count()-1))
        self.native(cf.ctypes, cb.ctypes, cs.ctypes, True, True, True,
                    self.c_ptr(Q0.ravel()), float(tf), self.c_ptr(nx), 1,
                    self.c_ptr(dx), float(cfl), self.c_ptr(boundaries), bool(stiff),
                    flux_code, int(order), 4, int(ndt), False,
                    self.c_ptr(ret.ravel()), threads)
        if not np.all(np.isfinite(ret)):
            raise FloatingPointError('Native solver returned a non-finite state; the segment was rejected.')
        return ret


def backend_information():
    """Record the library actually loaded; do not infer native success from imports."""
    import hashlib
    import platform
    import sys
    from pathlib import Path
    from ctypes import c_int
    from importlib.metadata import version
    from pypde.utils import get_cdll
    library = get_cdll()
    path = Path(library._name).resolve()
    safety = 0
    if hasattr(library, 'pypde_safety_api_version'):
        function = library.pypde_safety_api_version
        function.argtypes = []
        function.restype = c_int
        safety = int(function())
    nonconservative = 0
    if hasattr(library, 'pypde_nonconservative_api_version'):
        function = library.pypde_nonconservative_api_version
        function.argtypes = []
        function.restype = c_int
        nonconservative = int(function())
    return {'distribution': 'PyPDE', 'version': version('PyPDE'),
            'native_library': str(path), 'native_library_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'native_safety_api': safety, 'nonconservative_fix_api': nonconservative, 'python': sys.version.split()[0],
            'platform': platform.platform(), 'numpy': np.__version__,
            'numba': version('numba')}


def get_solver(cached=True):
    if cached:
        return CachedPyPDESolver()
    try:
        from pypde import pde_solver
    except ImportError as exc:
        raise ImportError('Install Haran Jackson PyPDE 1.0.0 in your existing working PyPDE environment.') from exc
    require_audited_backend(backend_information())
    return pde_solver


def require_audited_backend(info):
    """Do not silently run the known defective upstream B-gradient contraction."""
    if not info.get('nonconservative_fix_api') or not info.get('native_safety_api'):
        raise RuntimeError(
            'This two-layer model requires the audited native backend. Install '
            'vendor/PyPDE-1.0.0-tlpl-audited using tools/install_native_backend.sh. '
            'The uploaded upstream backend has incorrect B-gradient products '
            'and unsafe failure handling; importing pypde alone is not sufficient.')
