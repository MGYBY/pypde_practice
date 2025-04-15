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
    return (normal_depth*(1.0+dist_amp*np.sin(np.pi*2.0*x/wl)))

def u2_su(hr, ρr, μr):
    return ((3.0*μr+2.0*hr*ρr*(hr+3.0*μr))/(μr*(2.0+3.0*hr*ρr)))

def bss(hl, hu, ul, uu, μr):
    return ((12.0*hu*ul+18.0*hl*ul*μr-6.0*hl*uu*μr)/(4.0*hl*hu+3.0*hl**2.0*μr))

# problem-related parameters
fr_v = 0.80
hr_v = 1.0
μr_v = 1.0
ρr_v = 1.0
L_x = 5.0
dist_amp = 0.1

'''
hr_val = 1.0
mu_r_val = 0.70
rho_r_val = 0.70
L_x = 4.50
dist_amp = 0.020
tf = 12.0
'''

nx = 128
L = [L_x]
tf = 30.0
n_output = 30

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
    Q0[i, 2] = nv1*Q0[i, 0]
    Q0[i, 3] = nv2*Q0[i, 1]

# text files output
## output IC
file_name = "intial_condition"
with open(file_name, 'w') as f:
    i = 0
    out = np.zeros([1,nx,4])
    out[0,:,:] = Q0
    output_tot = np.zeros([1,6])
    for ix in range(0,nx):
        x = L_x/(nx*2.0)+ix*(L_x/nx)
        #output_tot = np.zeros((1, 8))
        output_tot[0,0] = x
        output_tot[0,1] = out[i,ix,0]
        output_tot[0,2] = out[i,ix,1]
        output_tot[0,3] = out[i,ix,2]
        output_tot[0,4] = out[i,ix,3]
        output_tot[0,5] = bss(nd1, nd2, nv1, nv2, μr_v)
        np.savetxt(f, output_tot, fmt='%g')

def F(Q):
    F_ = zeros(4)

    h1 = Q[0]
    h2 = Q[1]
    q1 = Q[2]
    q2 = Q[3]

    F_[0] = q1
    F_[1] = q2
    F_[2] = (96*h2**4*q1**2 + 6*h1*h2**2*q1*(21*h2*q1 + 2*h1*q2)*μr_v + 3*h1**2*(18*h2**2*q1**2 - 3*h1*h2*q1*q2 + 2*h1**2*q2**2)*μr_v**2)/(5.*h1*h2**2*(4*h2 + 3*h1*μr_v)**2) + (0.5/(fr_v**2.0))*h1**2.0
    F_[3] = q2**2/h2 + (4*h2*(3*h2*q1 - 2*h1*q2)**2)/(5.*h1**2*(4*h2 + 3*h1*μr_v)**2) + (0.5/(fr_v**2.0))*h2**2.0

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

    #ret[2, 0] = 0
    ret[2, 1] = ρr_v*(1.0/(fr_v*fr_v))*h1
    ret[3, 0] = (1.0/(fr_v*fr_v))*h2
    #ret[3, 1] = 0

    return ret


out = pde_solver(Q0, tf, L, F=F, S=S, B=B_dl, boundaryTypes='periodic', cfl=0.60, order=2, stiff=False, flux='rusanov', ndt=n_output, nThreads=1)


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
        wwriter.writerows(zip(np.transpose(x_array),np.transpose(out[i,:,0]),np.transpose(out[i,:,1]), (np.transpose(out[i,:,0])+np.transpose(out[i,:,1])),np.transpose(out[i,:,2]),np.transpose(out[i,:,3]), bss(out[i,:,0], out[i,:,1], out[i,:,2]/out[i,:,0], out[i,:,3]/out[i,:,1], μr_v)))
print("Finished field output. Beginning velocity-field reconstruction ... ...")

num_layers = 25 # number of sigma-layers in each layer
for i in range(0, frames):
    print('Reconstructing'+': '+str(i)+'th output')
    t = 1.0*i
    format_string_time = f"{t:.1f}"
    file_name = 'reconstructed_%s' % format_string_time
    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for cell in range(0,arr_size):
            x = L_x/(nx*2.0)+cell*(L_x/nx)
            hl = out[i,cell,0]
            hu = out[i,cell,1]
            ul = out[i,cell,2]/out[i,cell,0]
            uu = out[i,cell,3]/out[i,cell,1]
            c11 = (-6.0*hu*ul+9.0*hl*(-2.0*ul+uu)*μr_v)/(hl**2.0*(4.0*hu+3.0*hl*μr_v))
            c12 = (12.0*hu*ul+18.0*hl*ul*μr_v-6.0*hl*uu*μr_v)/(hl*(4.0*hu+3.0*hl*μr_v))
            c21 = (9.0*ul-6.0*uu)/(hu*(4.0*hu+3.0*hl*μr_v))
            c22 = (6.0*(hl+hu)*(2.0*uu-3.0*ul))/(hu*(4.0*hu+3.0*hl*μr_v))
            c23 = (3.0*(2.0*hu**2.0*ul+hl**2.0*(3.0*ul-2.0*uu)+hl*hu*(6.0*ul+uu*(-4.0+μr_v))))/(hu*(4.0*hu+3.0*hl*μr_v))
            # accomodate the boundary conditions at the wall
            output_tot = np.zeros((1, 3))
            output_tot[0,0] = x
            np.savetxt(f, output_tot, fmt='%g')
            # begin with lower layer
            for layer in range(0,num_layers):
                output_tot = np.zeros((1, 3))
                z = hl/(num_layers*2.0)+layer*(hl/num_layers)

                u = c11*z*z+c12*z
                output_tot[0,0] = x
                output_tot[0,1] = z
                output_tot[0,2] = u
                np.savetxt(f, output_tot, fmt='%g')
                if cell == 0:
                    output_tot[0,0] = 0.0
                    np.savetxt(f, output_tot, fmt='%g')
                if cell == (arr_size-1):
                    output_tot[0,0] = L_x
                    np.savetxt(f, output_tot, fmt='%g')
            # the upper layer
            for layer in range(0,num_layers):
                output_tot = np.zeros((1, 3))
                z = hl+hu/(num_layers*2.0)+layer*(hu/num_layers)
                u = c21*z*z+c22*z+c23
                output_tot[0,0] = x
                output_tot[0,1] = z
                output_tot[0,2] = u
                np.savetxt(f, output_tot, fmt='%g')
                if cell == 0:
                    output_tot[0,0] = 0.0
                    np.savetxt(f, output_tot, fmt='%g')
                if cell == (arr_size-1):
                    output_tot[0,0] = L_x
                    np.savetxt(f, output_tot, fmt='%g')
            # make sure the contour hits the free-surface
            z = hl+hu
            u = c21*z*z+c22*z+c23
            output_tot[0,0] = x
            output_tot[0,1] = z
            output_tot[0,2] = u
            np.savetxt(f, output_tot, fmt='%g')
            if cell == 0:
                    output_tot[0,0] = 0.0
                    np.savetxt(f, output_tot, fmt='%g')
            if cell == (arr_size-1):
                output_tot[0,0] = L_x
                np.savetxt(f, output_tot, fmt='%g')


