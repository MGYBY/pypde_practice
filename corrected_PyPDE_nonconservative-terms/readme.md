**An error found and corrected by Astra. All credit to GPT**

# PyPDE: nonconservative-term correction

This repository provides an independently patched version of [Haran Jackson's PyPDE](https://github.com/haranjackson/PyPDE), a Python/C++ solver based on ADER-WENO. It addresses a matrix–gradient multiplication error identified in the inspected PyPDE 1.0.0 source. Local package version: `1.0.0+tlpl.audit2`. **This is not an official upstream release.**

## What was the issue?

For a first-order, one-dimensional system,

$$
\frac{\partial\mathbf Q}{\partial t}
+\frac{\partial\mathbf F(\mathbf Q)}{\partial x}
+\mathbf B(\mathbf Q)\frac{\partial\mathbf Q}{\partial x}
=\mathbf S(\mathbf Q),
$$

the nonconservative term requires a matrix multiplied by a **column gradient**. For a system with `m` variables, this is an `(m × m) × (m × 1)` product.

PyPDE stores reconstructed gradients as rows. Two C++ statements attempted to multiply the matrix directly by a `(1 × m)` row. For `m > 1`, those dimensions are incompatible. With Eigen's assertions enabled, the tested expressions abort; with assertions disabled, they can silently produce incorrect contributions. The resulting behavior can depend on the compiler, Eigen version, and optimization settings.

## What was corrected?

**Galerkin predictor — `src/solvers/dg/dg.cpp`:**

```diff
- rett.row(idx) -= b * dq.row(idx * ndim + d);
+ rett.row(idx) -= (b * dq.row(idx * ndim + d).transpose()).transpose();
```

**Finite-volume cell update — `src/solvers/fv/fv.cpp`:**

```diff
- s -= b * dq[d].row(idx);
+ s -= b * dq[d].row(idx).transpose();
```

The inner transpose supplies the required gradient column. The outer transpose in the predictor returns the result to row storage. Both statements now evaluate the complete contraction

$$
(\mathbf B\mathbf Q_x)_i
=\sum_{j=1}^{m}B_{ij}\,\partial_x Q_j.
$$

**The user's `B(Q)` callback should not be transposed.** The separate nonconservative interface path term is retained; it does not replace the cell-interior contribution.

This correction follows Jackson's published formulation: Eq. (9b) in [Jackson (2017)](https://doi.org/10.1016/j.jcp.2016.12.058) defines the predictor contraction, while Eq. (49c) in his 2019 thesis defines the cell-interior integral. It corrects the implementation, not the underlying published method.

## Why does it matter?

Incorrect nonconservative coupling can change wave growth, propagation speed, and waveform shape—even when the simulation runs without crashing or conserves mass accurately. Uniform-flow tests cannot expose this error because the relevant spatial gradients vanish. Runs with `B=None` bypass the affected contractions.

The recorded local verification includes:

| Check | Original | Corrected |
|---|---:|---:|
| Two-layer complex Fourier-coefficient relative error, 256 cells | 2.09432% | 0.002061% |
| Periodic free-surface peak-to-trough height divided by lower-layer reference depth, 150 cells, `T = 24` | 1.170660 | 0.846611 |

The Fourier error decreases under refinement after correction, whereas the original error stalls near 2.1%. A build containing **only these two changes** reproduced the full patched distribution's periodic trajectory exactly at every saved time in the tested 150-cell case.

[Validation details and numerical records](VALIDATION.md) specify the test parameters, error definitions, source provenance, and limitations. These are **case-specific results**, not universal error estimates. The nonlinear profiles are not claimed to be saturated or grid-converged.

## Using this version

**Rebuild and reinstall the patched native C++ library.** Replacing only the Python simulation script does not apply this correction. Recheck earlier results that depend on nonzero nonconservative coupling, and perform application-specific convergence tests.

Build instructions are in [INSTALL.md](INSTALL.md). This is the native `pypde` library, not the two-layer application package.

The full patched distribution also contains separate error-handling fixes; [the complete native diff](evidence/full_native_patch.diff) records them, while the two-statement patch above isolates the numerical correction discussed here. This finding does not establish that every PyPDE release or Jackson's published simulations are affected.

## Credits

Original software: **Haran Jackson and contributors**. The original [AGPL license](LICENSE) and third-party notices are retained. See also the [upstream documentation](https://pypde.readthedocs.io/en/latest/) and [Eigen's matrix-arithmetic documentation](https://libeigen.gitlab.io/eigen/docs-3.3/group__TutorialMatrixArithmetic.html).
