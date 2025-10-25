from numba import njit
from numpy import array, zeros, power
from pypde import pde_solver

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os
import csv

import numpy as np
# initial conditions
def disturbed_area(x, normal_area, wl):
    return (normal_area*(1.0+dist_amp*np.sin(np.pi*2.0*x/wl)))

# problem-related parameters
n_v = 0.30
fr_v = 0.80
beta_v = 0.45
L_x = 25.0
dist_amp = 0.1
r0 = 1.0
q0 = 1.0

'''
hr_val = 1.0
mu_r_val = 0.70
rho_r_val = 0.70
L_x = 4.50
dist_amp = 0.020
tf = 12.0
'''

nx = 160
L = [L_x]
tf = 500.0
n_output = 10

delta_x = L_x/nx

Q0 = zeros([nx, 2])

for i in range(nx):
    x_coord = (i-0.50)*delta_x
    Q0[i, 0] = disturbed_area(x_coord, (np.pi*r0*r0), L_x)
    Q0[i, 1] = q0*disturbed_area(x_coord, (np.pi*r0*r0), L_x)/(np.pi*r0*r0)

# text files output
## output IC
file_name = "intial_condition"
with open(file_name, 'w') as f:
    i = 0
    out = np.zeros([1,nx,2])
    out[0,:,:] = Q0
    output_tot = np.zeros([1,3])
    for ix in range(0,nx):
        x = L_x/(nx*2.0)+ix*(L_x/nx)
        #output_tot = np.zeros((1, 8))
        output_tot[0,0] = x
        output_tot[0,1] = out[i,ix,0]
        output_tot[0,2] = out[i,ix,1]
        np.savetxt(f, output_tot, fmt='%g')

def F(Q):
    F_ = zeros(2)

    a = Q[0]
    q = Q[1]

    F_[0] = q
    F_[1] = ((3.0*n_v+1.0)*q*q)/((2.0*n_v+1.0)*a)+1.0/3.0*beta_v*(1.0/(np.pi**0.5))*(a**1.5)

    return F_

def S(Q):
    S_ = zeros(2)

    a = Q[0]
    q = Q[1]

    S_[0] = 0
    S_[1] = (((1.0/fr_v/fr_v)*(1.0/np.pi)*a/np.pi)-((q**n_v)/(np.pi*fr_v*fr_v*(((a/np.pi)**0.50)**(3.0*n_v-1.0)))))

    return S_

def B_dl(Q):
    ret = zeros((2, 2))
    return ret


out = pde_solver(Q0, tf, L, F=F, S=S, B=B_dl, boundaryTypes='periodic', cfl=0.850, order=2, stiff=False, flux='roe', ndt=n_output, nThreads=1)


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
        writer.writerows(zip(np.transpose(x_array),np.transpose(out[i,:,0]),np.transpose(out[i,:,1])))
print("Finished field output.")
