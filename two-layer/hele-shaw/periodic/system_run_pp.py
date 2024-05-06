from numba import njit
from numpy import array, zeros, power
from pypde import pde_solver

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os
import csv

##### steady-uniform velocity to make life easier #####
@njit
def u_su(h_r, rho_r, mu_r):
    return ((mu_r*(h_r*(4.0+3.0*h_r)+mu_r)+h_r*h_r*rho_r*(h_r*h_r+mu_r*(3.0+4.0*h_r)))/((1.0+h_r)*mu_r*(4.0*h_r+mu_r+3.0*h_r*h_r*rho_r)))

##### the beta1 term in advection term #####
@njit
def beta1_func(h_1, h_r, rho_r, mu_r):
    h_2 = 1.0+h_r-h_1
    numer = 6.0*(h_1**2.0*(16.0*h_2**2.0+7.0*h_1*h_2*mu_r+h_1**2.0*mu_r**2.0)+5.0*h_1*h_2**2.0*rho_r*(5.0*h_2+h_1*mu_r)+10.0*h_2**4.0*rho_r**2.0)
    denom = 5.0*(h_1*(4.0*h_2+h_1*mu_r)+3.0*h_2**2.0*rho_r)**2.0

    return (numer/denom)

##### the beta2 term in advection term #####
@njit
def beta2_func(h_1, h_r, rho_r, mu_r):
    h_2 = 1.0+h_r-h_1
    numer = 6.0*(h_2**2.0*rho_r**2.0*(h_2**2.0+7.0*h_1*h_2*mu_r+16*h_1**2*mu_r**2)+5.0*h_1**2*h_2*mu_r*rho_r*(h_2+5.0*h_1*mu_r)+10.0*h_1**4.0*mu_r**2.0)
    denom = 5.0*(h_2*rho_r*(h_2+4.0*h_1*mu_r)+3.0*h_1**2*mu_r)**2.0

    return (numer/denom)

##### the F1 term in source term #####
@njit
def F1_func(h_1, h_r, rho_r, mu_r):
    h_2 = 1.0+h_r-h_1
    numer = -1.0*12.0*(h_2+h_1*mu_r)
    denom = h_1*(4.0*h_2+h_1*mu_r)+3.0*rho_r*h_2**2.0

    return (numer/denom)

##### the F2 term in source term #####
@njit
def F2_func(h_1, h_r, rho_r, mu_r):
    h_2 = 1.0+h_r-h_1
    numer = -1.0*12.0*rho_r*(h_2+h_1*mu_r)
    denom = 3.0*h_1**2.0*mu_r+h_2*rho_r*(h_2+4.0*h_1*mu_r)

    return (numer/denom)

##### modified compatibility condition -- relation between (1/(So*Re)) and others (Fr, h_r, rho_r, mu_r) #####
@njit
def sore_reci(fr, h_r, rho_r, mu_r):
    return (1.0/(fr**2)*(4.0*h_r+mu_r+3.0*h_r*h_r*rho_r)/(12.0*(h_r+mu_r)))

# problem-related parameters
fr_val = 1.00
hr_val = 10.0
mu_r_val = 0.02
rho_r_val = 0.001
L_x = 4.50
dist_amp = 0.020

'''
hr_val = 1.0
mu_r_val = 0.70
rho_r_val = 0.70
L_x = 4.50
dist_amp = 0.020
tf = 12.0
'''

def F(Q):
    F_ = zeros(2)

    h = Q[0]
    q = Q[1]

    F_[0] = q
    F_[1] = 1.0/(1.0+rho_r_val)*(beta1_func(h, hr_val, rho_r_val, mu_r_val)*q**2.0/h - beta2_func(h, hr_val, rho_r_val, mu_r_val)*((u_su(hr_val, rho_r_val, mu_r_val)*(1.0+hr_val)-q)/(1.0+hr_val-h))**2.0*(1.0+hr_val-h) + (1.0-rho_r_val)*0.50*(1.0/(fr_val**2.0))*h**2.0)

    return F_

def S(Q):
    S_ = zeros(2)

    h = Q[0]
    q = Q[1]

    S_[1] = 1.0/(1.0+rho_r_val)*((1.0-rho_r_val*hr_val)*(1.0/(fr_val**2.0)) + sore_reci(fr_val, hr_val, rho_r_val, mu_r_val)*(q/h*F1_func(h, hr_val, rho_r_val, mu_r_val)-mu_r_val*((u_su(hr_val, rho_r_val, mu_r_val)*(1.0+hr_val)-q)/(1.0+hr_val-h)*F2_func(h, hr_val, rho_r_val, mu_r_val))))

    return S_

import numpy as np
# initial conditions
def disturbed_depth(x, normal_depth, wl):
    return (normal_depth*(1.0+dist_amp*np.sin(np.pi*2.0*x/wl)))

nx = 200
L = [L_x]
tf = 20.0

delta_x = L_x/nx
nd = 1.0
nv = 1.0

Q0 = zeros([nx, 2])

for i in range(nx):
    x_coord = (i-0.50)*delta_x
    Q0[i, 0] = disturbed_depth(x_coord, nd, L_x)
    #Q0[i, 1] = u_su(hr_val, rho_r_val, mu_r_val)*nd
    Q0[i, 1] = nv*nd

out = pde_solver(Q0, tf, L, F=F, S=S, boundaryTypes='periodic', cfl=0.240, order=2, stiff=False, flux='rusanov', ndt=51, nThreads=4)


# text files output
import numpy as np
un_shape = np.shape(out)
frames = un_shape[0]
arr_size = np.size(out[0,:,0])
x_array = np.linspace(0, L_x, num=arr_size)
for i in range(0, frames):
    t = 1.0*i
    format_string_time = f"{t:.1f}"
    file_name = 'outXYZ_%s' % format_string_time
    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(np.transpose(x_array),np.transpose(out[i,:,0])))
