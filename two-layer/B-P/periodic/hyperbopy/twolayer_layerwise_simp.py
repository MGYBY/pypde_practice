"""
Here we solve the following system of equations:
    - d[h1]/dt + d[q1]/dx = 0
    - d[q1]/dt + d[q1**2/h1 + (g/2)*h1]/dx = -g*h1*(d[h2 + Z]/dx)
    - d[h2]/dt + d[q2]/dt = 0
    - d[q2]/dt + d[q2**2/h2 + (g/2)*h2**2]/dx = -g*h2*(d[r*h1 + Z]/dx)

with:
    - r = rho1/rho2 

variables:
    - U  = [h1, q1, h2, q2]:
    - W = [h1, q1, h2, q2, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (5, Nx)

dim(W_int) = (5, 2, Nx):
    - 2: [h1, q1, h2, q2, Z]
    - 2: [pos, min]

REFERENCE: Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

"""

import numpy as np

from .basemodel import BaseModel


class SW2LLayerwise_simp(BaseModel):

    name = 'SW2LLayerwise_simp'

    def __init__(self, g=None, r=None, theta=None, epsilon=None, dt_fact=None):
        self.g = g if g is not None else self.GRAVITATIONAL_CONSTANT
        self.r = r if r is not None else self.DENSITY_RATIO
        self.theta = theta if theta is not None else self.THETA
        self.epsilon = epsilon if epsilon is not None else self.EPSILON
        self.dt_fact = dt_fact if dt_fact is not None else self.DT_FACT
        #
        self.var_names = ['h1', 'q1', 'h2', 'q2', 'Z']

    # #### spatial discretization functions

    def compute_F(self, W_int):
        return np.swapaxes(
            np.array([W_int[1, ...],
                      W_int[1, ...]**2/W_int[0, ...] +
                          (self.g/2)*W_int[0, ...]**2,
                      W_int[3, ...],
                      W_int[3, ...]**2/W_int[2, ...] +
                          (self.g/2)*W_int[2, ...]**2,
                      ]),
            0, 1)

    def compute_S(self, W, W_int):
        l1 = -self.g*W[0, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
        l2 = -self.g*W[2, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
        return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])

    def compute_B(self, W, W_int):
        l1 = -self.g*W[0, 1:-1]*(W_int[2, 1, 1:] - W_int[2, 0, :-1])
        l2 = -self.g*self.r*W[2, 1:-1]*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
        return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])

    def compute_Spsi_int(self, W, W_int):
        l1 = -(self.g/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
            (W_int[-1, 0, :] - W_int[-1, 1, :])
        l2 = -(self.g/2)*(W_int[2, 0, :] + W_int[2, 1, :]) * \
            (W_int[-1, 0, :] - W_int[-1, 1, :])
        return np.array([np.zeros_like(l1), np.zeros_like(l1), np.zeros_like(l1), np.zeros_like(l2)])

    def compute_Bpsi_int(self, W, W_int):
        l1 = -(self.g/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
            (W_int[2, 0, :] - W_int[2, 1, :])
        l2 = -(self.g*self.r/2)*(W_int[2, 0, :] + W_int[2, 1, :]) * \
            (W_int[0, 0, :] - W_int[0, 1, :])
        return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])

    def compute_Ainv_int(self, W, W_int):
        zero = np.zeros_like(W_int[0, 0, :])
        one = np.ones_like(W_int[0, 0, :])
        #
        l1 = np.array([zero, zero, zero, zero])
        l2 = np.array([zero, zero, zero, zero])
        l3 = np.array([zero, zero, zero, zero])
        l4 = np.array([zero, zero, zero, zero])
        return np.array([l1, l2, l3, l4])

    def compute_local_speeds(self, W_int, dx):
        h2_int = W_int[2, ...] - W_int[4, ...]
        um = (W_int[1, ...] + W_int[3, ...])/(W_int[0, ...] + h2_int)
        #
        ap_int = np.row_stack(
            (um + np.sqrt(self.g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).max(axis=0)
        am_int = np.row_stack(
            (um - np.sqrt(self.g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).min(axis=0)
        return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))
