from numba import njit
import matplotlib.pyplot as plt
from numpy import array, zeros, power
from pypde import pde_solver

##### steady-uniform velocity to make life easier #####
@njit
def u_su(h_r, rho_r, mu_r):
    return ((mu_r*(h_r*(4.0+3.0*h_r)+mu_r)+h_r*h_r*rho_r*(h_r*h_r+mu_r*(3.0+4.0*h_r)))/((1.0+h_r)*mu_r*(4.0*h_r+mu_r+3.0*h_r*h_r*mu_r)))

##### the beta1 term in advection term #####
@njit
def beta1_func(h_1, h_r, rho_r, mu_r):
    h_2 = 1.0+h_r-h_1
    return (6*(Power(h1,2)*(16*Power(h2,2)*Power(mu1,2) + 7*h1*h2*mu1*mu2 + Power(h1,2)*Power(mu2,2))*Power(\[Rho]1,2) + 5*h1*Power(h2,2)*mu1*(5*h2*mu1 + h1*mu2)*\[Rho]1*\[Rho]2 + 10*Power(h2,4)*Power(mu1,2)*Power(\[Rho]2,2)))/(5.*Power(h1*(4*h2*mu1 + h1*mu2)*\[Rho]1 + 3*Power(h2,2)*mu1*\[Rho]2,2))
