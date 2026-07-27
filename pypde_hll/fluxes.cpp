#include "fluxes.h"
#include "../../eigs/system.h"
#include "../../types.h"
#include "eigen3/Eigenvalues"
#include <cmath>

FluxGenerator::FluxGenerator(void (*_F)(double *, double *, double *, int),
                             void (*_B)(double *, double *, int), Vecr _NODES,
                             Vecr _WGHTS, Vecr _dX, int _V, int _FLUX, int ndim)
    : F(_F), B(_B), NODES(_NODES), WGHTS(_WGHTS), dX(_dX), V(_V), FLUX(_FLUX) {

  N = NODES.size();
  fL = Vec(V);
  fR = Vec(V);
  q = Vec(V);
  dq = Mat(ndim, V);
  M = Mat(V, V);
}

Vec FluxGenerator::D_OSH(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d) {
  // Returns the Osher flux component, in the dth direction

  Vec Dq = qL - qR;
  Mat Ddq = dqL - dqR;

  cVec Dqc = cVec(Dq);

  cVec b(V);
  cVec ret = cVec::Zero(V);

  for (int i = 0; i < N; i++) {

    q = qR + NODES(i) * Dq;
    dq = dqR + NODES(i) * Ddq;

    M = system_matrix(F, B, q, dq, d);
    ES.compute(M);

    b = ES.eigenvectors().colPivHouseholderQr().solve(Dqc).array() *
        ES.eigenvalues().array().abs();
    ret += WGHTS(i) * (ES.eigenvectors() * b);
  }

  return ret.real();
}

Vec FluxGenerator::D_ROE(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d) {
  // Returns the Osher flux component, in the dth direction

  Vec Dq = qL - qR;
  Mat Ddq = dqL - dqR;

  cVec Dqc = cVec(Dq);

  M.setZero();

  for (int i = 0; i < N; i++) {

    q = qR + NODES(i) * Dq;
    dq = dqR + NODES(i) * Ddq;

    M += WGHTS(i) * system_matrix(F, B, q, dq, d);
  }
  ES.compute(M);

  cVec b = ES.eigenvectors().colPivHouseholderQr().solve(Dqc).array() *
           ES.eigenvalues().array().abs();

  return (ES.eigenvectors() * b).real();
}

Vec FluxGenerator::D_RUS(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d) {

  double max1 = max_abs_eigs(F, B, qL, dqL, d);
  double max2 = max_abs_eigs(F, B, qR, dqR, d);

  return std::max(max1, max2) * (qL - qR);
}


void FluxGenerator::hll_fluxes(Vecr retL, Vecr retR, Vecr qL, Vecr qR,
                               Matr dqL, Matr dqR, int d,
                               bool secondOrder) {
  // Path-conservative two-wave HLL solver.  For B == NULL this reduces to the
  // standard conservative HLL flux.  For B != NULL it uses the same straight
  // state-space path and quadrature as Bint().

  // The endpoint-gradient work arrays in fv.cpp are intentionally not filled
  // for first-order fluxes.  Avoid reading those uninitialised values here.
  Mat dqLsafe = secondOrder ? Mat(dqL) : Mat::Zero(dqL.rows(), dqL.cols());
  Mat dqRsafe = secondOrder ? Mat(dqR) : Mat::Zero(dqR.rows(), dqR.cols());

  if (F == NULL) {
    fL.setZero();
    fR.setZero();
  } else {
    F(fL.data(), qL.data(), dqLsafe.data(), d);
    F(fR.data(), qR.data(), dqRsafe.data(), d);
  }

  Vec pathJump = Vec::Zero(V);
  if (B != NULL)
    Bint(pathJump, qL, qR, d);

  const Vec dQ = qR - qL;
  const Vec G = fR - fL + pathJump;

  double sL = 0.;
  double sR = 0.;
  double rho = 0.;
  bool realSpectrum = true;

  auto sample_bounds = [&](Vecr qs, Matr dqs) {
    Mat A = system_matrix(F, B, qs, dqs, d);
    double lo, hi;
    if (!eigenvalue_bounds(A, lo, hi))
      realSpectrum = false;
    sL = std::min(sL, lo);
    sR = std::max(sR, hi);
    rho = std::max(rho, _max_abs_eig(A));
  };

  // End-point estimates are inexpensive and guarantee that both reconstructed
  // states are included.  The quadrature samples make the bound more robust
  // for nonlinear systems and are consistent with PyPDE's straight path.
  sample_bounds(qL, dqLsafe);
  sample_bounds(qR, dqRsafe);

  const Vec Dq = qL - qR;
  const Mat Ddq = dqLsafe - dqRsafe;
  for (int i = 0; i < N; i++) {
    q = qR + NODES(i) * Dq;
    dq = dqR + NODES(i) * Ddq;
    sample_bounds(q, dq);
  }

  // Ordered HLL speeds do not exist after loss of hyperbolicity.  Preserve a
  // robust numerical fallback by reverting locally to symmetric Rusanov
  // bounds.  Production applications should still monitor complex
  // eigenvalues independently.
  if (!realSpectrum) {
    sL = -rho;
    sR = rho;
  }

  Vec dMinus = Vec::Zero(V);
  Vec dPlus = Vec::Zero(V);

  if (sL >= 0.) {
    // All waves move to the right.
    dPlus = G;
  } else if (sR <= 0.) {
    // All waves move to the left.
    dMinus = G;
  } else {
    const double denom = sR - sL;
    if (denom <= 100. * mEPS) {
      // Degenerate zero-speed fan.  A symmetric split avoids division by zero.
      dMinus = 0.5 * G;
      dPlus = 0.5 * G;
    } else {
      dMinus = sL / denom * (sR * dQ - G);
      dPlus = sR / denom * (G - sL * dQ);
    }
  }

  // The two interface contributions differ by the path jump.  For a
  // conservative system they coincide and are equal to the ordinary HLL flux.
  retL = 2. * (fL + dMinus);
  retR = 2. * (fR - dPlus);

  // Retain PyPDE's existing local Lax-Friedrichs stabilization for
  // gradient-dependent (parabolic) fluxes.
  if (secondOrder) {
    const double max1 =
        max_abs_eigs_second_order(F, qL, dqL, d, N, dX);
    const double max2 =
        max_abs_eigs_second_order(F, qR, dqR, d, N, dX);
    const Vec penalty = std::max(max1, max2) * (qL - qR);
    retL += penalty;
    retR += penalty;
  }
}

void FluxGenerator::flux(Vecr ret, Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d,
                         bool secondOrder) {

  if (FLUX == RUSANOV)
    ret = D_RUS(qL, qR, dqL, dqR, d);

  if (FLUX == ROE)
    ret = D_ROE(qL, qR, dqL, dqR, d);

  if (FLUX == OSHER)
    ret = D_OSH(qL, qR, dqL, dqR, d);

  F(fL.data(), qL.data(), dqL.data(), d);
  F(fR.data(), qR.data(), dqR.data(), d);
  ret += fL + fR;

  if (secondOrder) {
    double max1 = max_abs_eigs_second_order(F, qL, dqL, d, N, dX);
    double max2 = max_abs_eigs_second_order(F, qR, dqR, d, N, dX);
    ret += std::max(max1, max2) * (qL - qR);
  }
}

void FluxGenerator::Bint(Vecr ret, Vecr qL, Vecr qR, int d) {
  // Returns the jump matrix for B, in the dth direction

  Vec Dq = qR - qL;

  Mat b(V, V);
  M.setZero();

  for (int i = 0; i < N; i++) {
    q = qL + NODES(i) * Dq;
    B(b.data(), q.data(), d);
    M += WGHTS(i) * b;
  }
  ret = M * Dq;
}