#ifndef SYSTEM_H
#define SYSTEM_H

#include "../types.h"

Mat system_matrix(void (*F)(double *, double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, Matr dq, int d);

double max_abs_eigs(void (*F)(double *, double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, Matr dq, int d);

// Returns the smallest and largest real parts of the eigenvalues of M.
// The boolean return value is false when a non-negligible imaginary part is
// detected.  This is used by the HLL solver, which requires ordered real wave
// speeds rather than only a spectral radius.
bool eigenvalue_bounds(Matr M, double &minEig, double &maxEig);

double _max_abs_eig(Matr M);

double max_abs_eigs_second_order(void (*F)(double *, double *, double *, int),
                                 Vecr q, Matr dq, int d, int N, Vecr dX);

#endif
