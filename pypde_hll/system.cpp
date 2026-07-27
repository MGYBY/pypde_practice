#include "../types.h"
#include "CustomMatProd.h"
#include "NumericalDiff.h"
#include "eigen3/Eigenvalues"

Mat system_matrix(void (*F)(double *, double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, Matr dq, int d) {

  int V = q.size();

  Mat M(V, V);

  if (F == NULL)
    M.setZero();
  else
    df(M, F, q, dq, d, false);

  if (B != NULL) {

    Mat b(V, V);
    B(b.data(), q.data(), d);
    M += b;
  }

  return M;
}

double _max_abs_eig(Matr M) {

  int V = M.rows();

  // TODO: work out when it is optimal to use Spectra
  // (presumably for larger matrices)
  if (V < 6)
    return M.eigenvalues().array().abs().maxCoeff();

  CustomMatProd op(M);
  SpectraEigSolver eigs(&op, 1, V);
  eigs.init();
  eigs.compute();

  return std::abs(eigs.eigenvalues()(0));
}


bool eigenvalue_bounds(Matr M, double &minEig, double &maxEig) {

  Eigen::EigenSolver<Mat> eigs(M, false);
  cVec vals = eigs.eigenvalues();

  minEig = INF;
  maxEig = -INF;

  double scale = 1.;
  for (int i = 0; i < vals.size(); i++)
    scale = std::max(scale, std::abs(vals(i)));

  bool realSpectrum = true;
  const double imagTol = 1.e-10 * scale;

  for (int i = 0; i < vals.size(); i++) {
    minEig = std::min(minEig, vals(i).real());
    maxEig = std::max(maxEig, vals(i).real());
    if (std::abs(vals(i).imag()) > imagTol)
      realSpectrum = false;
  }

  return realSpectrum;
}

double max_abs_eigs(void (*F)(double *, double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, Matr dq,
                    int d) {

  Mat M = system_matrix(F, B, q, dq, d);
  return _max_abs_eig(M);
}

double max_abs_eigs_second_order(void (*F)(double *, double *, double *, int),
                                 Vecr q, Matr dq, int d, int N, Vecr dX) {

  int V = q.size();
  Mat M2(V, V);

  df(M2, F, q, dq, d, true);
  return 2 * (N + 1) / dX(d) * _max_abs_eig(M2);
}
