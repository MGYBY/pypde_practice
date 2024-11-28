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
def disturbed_depth(x, normal_depth, wl):
    if (x>=(disp_wl/2.0) and x<=disp_wl):
        return (normal_depth*(1.0+dist_amp*np.sin(np.pi*2.0*x/wl)))
    else:
        return (normal_depth)

def u2_su(hr, ρr, μr):
    return ((3.0*μr+2.0*hr*ρr*(hr+3.0*μr))/(μr*(2.0+3.0*hr*ρr)))

# problem-related parameters
fr_v = 0.60
hr_v = 0.50
μr_v = 0.33333
ρr_v = 0.98681
L_x = 75.0
dist_amp = 0.20
disp_wl = 2.0

'''
hr_val = 1.0
mu_r_val = 0.70
rho_r_val = 0.70
L_x = 4.50
dist_amp = 0.020
tf = 12.0
'''

nx = 2100
L = [L_x]
tf = 24.0
n_output = 48

delta_x = L_x/nx
nd1 = 1.0
nd2 = nd1*hr_v
nv1 = 1.0
nv2 = u2_su(hr_v, ρr_v, μr_v)

Q0 = zeros([nx, 4])

for i in range(nx):
    x_coord = (i-0.50)*delta_x
    Q0[i, 0] = disturbed_depth(x_coord, nd1, L_x)
    Q0[i, 1] = disturbed_depth(x_coord, nd2, L_x)
    Q0[i, 2] = nv1*((disturbed_depth(x_coord, nd1, L_x)/nd1)**0.50)*Q0[i, 0]
    Q0[i, 3] = nv2*((disturbed_depth(x_coord, nd2, L_x)/nd2)**0.50)*Q0[i, 1]

def F(Q):
    F_ = zeros(4)

    h1 = Q[0]
    h2 = Q[1]
    q1 = Q[2]
    q2 = Q[3]

    F_[0] = q1
    F_[1] = q2
    F_[2] = (96*h2**4*q1**2 + 6*h1*h2**2*q1*(21*h2*q1 + 2*h1*q2)*μr_v + 3*h1**2*(18*h2**2*q1**2 - 3*h1*h2*q1*q2 + 2*h1**2*q2**2)*μr_v**2)/(5.*h1*h2**2*(4*h2 + 3*h1*μr_v)**2)
    F_[3] = q2**2/h2 + (4*h2*(3*h2*q1 - 2*h1*q2)**2)/(5.*h1**2*(4*h2 + 3*h1*μr_v)**2)

    return F_

def S(Q):
    S_ = zeros(4)

    h1 = Q[0]
    h2 = Q[1]
    q1 = Q[2]
    q2 = Q[3]

    S_[2] = ((6*(-2*h2**2*q1 - 6*h1*h2*q1*μr_v + 3*h1**2*q2*μr_v))/(h1**2*h2*(4*h2 + 3*h1*μr_v)))*(1.0/(6.0*fr_v*fr_v/(2.0+3.0*hr_v*ρr_v)))+(1.0/(fr_v*fr_v))*h1
    S_[3] = ((18*h2*q1*μr_v - 12*h1*q2*μr_v)/(4*h1*h2**2 + 3*h1**2*h2*μr_v))*(1.0/(6.0*fr_v*fr_v*ρr_v/(2.0+3.0*hr_v*ρr_v)))+(1.0/(fr_v*fr_v))*h2

    return S_

def B_dl(Q):
    ret = zeros((4, 4))

    h1 = Q[0]
    h2 = Q[1]
    q1 = Q[2]
    q2 = Q[3]

    ret[2, 0] = (1.0/(fr_v*fr_v))*h1
    ret[2, 1] = ρr_v*(1.0/(fr_v*fr_v))*h1
    ret[3, 0] = (1.0/(fr_v*fr_v))*h2
    ret[3, 1] = (1.0/(fr_v*fr_v))*h2

    return ret


out = pde_solver(Q0, tf, L, F=F, S=S, B=B_dl, boundaryTypes='periodic', cfl=0.750, order=2, stiff=False, flux='rusanov', ndt=n_output, nThreads=6)


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
        writer.writerows(zip(np.transpose(x_array),np.transpose(out[i,:,0]),np.transpose(out[i,:,1]), (np.transpose(out[i,:,0])+np.transpose(out[i,:,1]))))
