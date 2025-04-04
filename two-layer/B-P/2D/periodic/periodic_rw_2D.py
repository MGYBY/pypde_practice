from numba import njit
from numpy import array, zeros, power
from pypde import pde_solver, weno_solver

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os
import csv

import numpy as np
# initial conditions
def disturbed_depth(x, y, normal_depth, wl_x, wl_y):
    return (normal_depth*(1.0+dist_amp_x*np.sin(np.pi*2.0*x/wl_x)+dist_amp_y*np.sin(np.pi*2.0*y/wl_y)))

def u2_su(hr, ρr, μr):
    return ((3.0*μr+2.0*hr*ρr*(hr+3.0*μr))/(μr*(2.0+3.0*hr*ρr)))

# problem-related parameters
fr_v = 1.0
hr_v = 1.0
μr_v = 1.0
ρr_v = 1.0
L_x = 15.0
L_y = 15.0
dist_amp_x = 0.1
dist_amp_y = 0.1
dist_wl_x = 5.0
dist_wl_y = 5.0

'''
hr_val = 1.0
mu_r_val = 0.70
rho_r_val = 0.70
L_x = 4.50
dist_amp = 0.020
tf = 12.0
'''

nx = 256
ny = 256
L = [L_x, L_y]
tf = 32.0
n_output = 32

delta_x = L_x/nx
delta_y = L_y/ny
nd1 = 1.0
nd2 = nd1*hr_v
nv1 = 1.0
nv2 = u2_su(hr_v, ρr_v, μr_v)
'''
must need additional variables to avoid
"erminate called after throwing an instance of 'std::runtime_error'
  what():  UpperHessenbergEigen: eigen decomposition failed
" errors (why?)
'''
num_var = 11

Q0 = zeros([nx, ny, num_var])

for i in range(nx):
    for j in range(ny):
        x_coord = (i-0.50)*delta_x
        y_coord = (j-0.50)*delta_y
        Q0[i, j, 0] = disturbed_depth(x_coord, y_coord, nd1, dist_wl_x, dist_wl_y)
        Q0[i, j, 1] = disturbed_depth(x_coord, y_coord, nd2, dist_wl_x, dist_wl_y)
        # constant-Fr setups
        #Q0[i, j, 2] = nv1*(Q0[i, j, 0]/(nd1))**0.50
        #Q0[i, j, 3] = nv2*(Q0[i, j, 1]/(nd2))**0.50
        Q0[i, j, 2] = nv1*(nd1)
        Q0[i, j, 3] = nv2*(nd2)
        Q0[i, j, 4] = 0.0
        Q0[i, j, 5] = 0.0

def F(Q,d):
    F_ = zeros(num_var)

    h1 = Q[0]
    h2 = Q[1]
    q1x = Q[2]
    q2x = Q[3]
    q1y = Q[4]
    q2y = Q[5]

    if d==0:
        ## x-direction
        F_[0] = q1x
        F_[1] = q2x
        # F1xx
        F_[2] = (96*h2**4*q1x**2 + 6*h1*h2**2*q1x*(21*h2*q1x + 2*h1*q2x)*μr_v + 3*h1**2*(18*h2**2*q1x**2 - 3*h1*h2*q1x*q2x + 2*h1**2*q2x**2)*μr_v**2)/(5.*h1*h2**2*(4*h2 + 3*h1*μr_v)**2) + (0.5/(fr_v**2.0))*h1**2.0
        # F2xx
        F_[3] = q2x**2/h2 + (4*h2*(3*h2*q1x - 2*h1*q2x)**2)/(5.*h1**2*(4*h2 + 3*h1*μr_v)**2) + (0.5/(fr_v**2.0))*h2**2.0
        # F1yx
        F_[4] = (3*(64*h2**4*q1x*q1y + 4*h1*h2**2*(21*h2*q1x*q1y + h1*q1y*q2x + h1*q1x*q2y)*μr_v + h1**2*(36*h2**2*q1x*q1y + 4*h1**2*q2x*q2y - 3*h1*h2*(q1y*q2x + q1x*q2y))*μr_v**2))/(10.*h1*h2**2*(4*h2 + 3*h1*μr_v)**2)
        # F2yx
        F_[5] = (q2x*q2y)/h2 + (4*h2*(-3*h2*q1x + 2*h1*q2x)*(-3*h2*q1y + 2*h1*q2y))/(5.*h1**2*(4*h2 + 3*h1*μr_v)**2)
    else:
        ## y-direction
        F_[0] = q1y
        F_[1] = q2y
        # F1xy
        F_[2] = (3*(64*h2**4*q1x*q1y + 4*h1*h2**2*(21*h2*q1x*q1y + h1*q1y*q2x + h1*q1x*q2y)*μr_v + h1**2*(36*h2**2*q1x*q1y + 4*h1**2*q2x*q2y - 3*h1*h2*(q1y*q2x + q1x*q2y))*μr_v**2))/(10.*h1*h2**2*(4*h2 + 3*h1*μr_v)**2)
        # F2xy
        F_[3] = (q2x*q2y)/h2 + (4*h2*(-3*h2*q1x + 2*h1*q2x)*(-3*h2*q1y + 2*h1*q2y))/(5.*h1**2*(4*h2 + 3*h1*μr_v)**2)
        # F1yy
        F_[4] = (96*h2**4*q1y**2 + 6*h1*h2**2*q1y*(21*h2*q1y + 2*h1*q2y)*μr_v + 3*h1**2*(18*h2**2*q1y**2 - 3*h1*h2*q1y*q2y + 2*h1**2*q2y**2)*μr_v**2)/(5.*h1*h2**2*(4*h2 + 3*h1*μr_v)**2) + (0.5/(fr_v**2.0))*h1**2.0
        # F2yy
        F_[5] = q2y**2/h2 + (4*h2*(3*h2*q1y - 2*h1*q2y)**2)/(5.*h1**2*(4*h2 + 3*h1*μr_v)**2) + (0.5/(fr_v**2.0))*h2**2.0


    return F_

def S(Q):
    S_ = zeros(num_var)

    h1 = Q[0]
    h2 = Q[1]
    q1x = Q[2]
    q2x = Q[3]
    q1y = Q[4]
    q2y = Q[5]

    S_[2] = ((6*(-2*h2**2*q1x - 6*h1*h2*q1x*μr_v + 3*h1**2*q2x*μr_v))/(h1**2*h2*(4*h2 + 3*h1*μr_v)))*(1.0/(6.0*fr_v*fr_v/(2.0+3.0*hr_v*ρr_v)))+(1.0/(fr_v*fr_v))*h1
    S_[3] = ((18*h2*q1x*μr_v - 12*h1*q2x*μr_v)/(4*h1*h2**2 + 3*h1**2*h2*μr_v))*(1.0/(6.0*fr_v*fr_v*ρr_v/(2.0+3.0*hr_v*ρr_v)))+(1.0/(fr_v*fr_v))*h2
    S_[4] = ((6*(-2*h2**2*q1y - 6*h1*h2*q1y*μr_v + 3*h1**2*q2y*μr_v))/(h1**2*h2*(4*h2 + 3*h1*μr_v)))*(1.0/(6.0*fr_v*fr_v/(2.0+3.0*hr_v*ρr_v)))
    S_[5] = ((18*h2*q1y*μr_v - 12*h1*q2y*μr_v)/(4*h1*h2**2 + 3*h1**2*h2*μr_v))*(1.0/(6.0*fr_v*fr_v*ρr_v/(2.0+3.0*hr_v*ρr_v)))

    return S_

def B_dl(Q,d):
    ret = zeros((num_var, num_var))

    h1 = Q[0]
    h2 = Q[1]
    q1x = Q[2]
    q2x = Q[3]
    q1y = Q[4]
    q2y = Q[5]

    if d==0:
        #ret[2, 0] = 0
        ret[2, 1] = ρr_v*(1.0/(fr_v*fr_v))*h1
        ret[4, 0] = (1.0/(fr_v*fr_v))*h2
        #ret[3, 1] = 0
    else:
        ret[3, 1] = ρr_v*(1.0/(fr_v*fr_v))*h1
        ret[5, 0] = (1.0/(fr_v*fr_v))*h2

    return ret


out = pde_solver(Q0, tf, L, F=F, S=S, B=B_dl, boundaryTypes='periodic', cfl=0.825, order=1, stiff=False, flux='rusanov', ndt=n_output, nThreads=6)

# text files output
# FIXME: here is written out of imagination, please double check later.
import numpy as np
un_shape = np.shape(out)
frames = un_shape[0]
#arr_size = np.size(out[0,:,0])
#x_array = np.linspace(0, L_x, num=arr_size)
#for i in range(0, frames):
    #t = 1.0*i
    #format_string_time = f"{t:.1f}"
    #file_name = 'outXYZ_%s' % format_string_time
    #with open(file_name, 'w') as f:
        #writer = csv.writer(f, delimiter='\t')
        #writer.writerows(zip(np.transpose(x_array),np.transpose(out[i,:,0]),np.transpose(out[i,:,1]), (np.transpose(out[i,:,0])+np.transpose(out[i,:,1])),np.transpose(out[i,:,2]),np.transpose(out[i,:,3])))

# FIXME: it may need a faster output routine (using .reshape(-1) methods)
output_tot = np.zeros((1, 8))
for i in range(0, frames):
    print('Outputting'+': '+str(i)+'th 3D output')
    t = 1.0*i
    format_string_time = f"{t:.1f}"
    file_name = 'output_%s' % format_string_time
    with open(file_name, 'w') as f:
        for ix in range(0,nx):
            for iy in range(0,ny):
                x = L_x/(nx*2.0)+ix*(L_x/nx)
                y = L_y/(ny*2.0)+iy*(L_y/ny)
                #output_tot = np.zeros((1, 8))
                output_tot[0,0] = x
                output_tot[0,1] = y
                output_tot[0,2] = out[i,ix,iy,0]
                output_tot[0,3] = out[i,ix,iy,1]
                output_tot[0,4] = out[i,ix,iy,2]
                output_tot[0,5] = out[i,ix,iy,3]
                output_tot[0,6] = out[i,ix,iy,4]
                output_tot[0,7] = out[i,ix,iy,5]
                np.savetxt(f, output_tot, fmt='%g')

        # Consideration of the "boundary" points
        for ix in range(0,nx):
            x = L_x/(nx*2.0)+ix*(L_x/nx)
            #output_tot = np.zeros((1, 8))
            output_tot[0,0] = x
            output_tot[0,1] = 0
            output_tot[0,2] = out[i,ix,0,0]-(out[i,ix,1,0]-out[i,ix,0,0])/2.0
            output_tot[0,3] = out[i,ix,0,1]-(out[i,ix,1,1]-out[i,ix,0,1])/2.0
            output_tot[0,4] = out[i,ix,0,2]-(out[i,ix,1,2]-out[i,ix,0,2])/2.0
            output_tot[0,5] = out[i,ix,0,3]-(out[i,ix,1,3]-out[i,ix,0,3])/2.0
            output_tot[0,6] = out[i,ix,0,4]-(out[i,ix,1,4]-out[i,ix,0,4])/2.0
            output_tot[0,7] = out[i,ix,0,5]-(out[i,ix,1,5]-out[i,ix,0,5])/2.0
            np.savetxt(f, output_tot, fmt='%g')
            #output_tot[0,0] = x
            output_tot[0,1] = L_y
            output_tot[0,2] = out[i,ix,-1,0]+(out[i,ix,-1,0]-out[i,ix,-2,0])/2.0
            output_tot[0,3] = out[i,ix,-1,1]+(out[i,ix,-1,1]-out[i,ix,-2,1])/2.0
            output_tot[0,4] = out[i,ix,-1,2]+(out[i,ix,-1,2]-out[i,ix,-2,2])/2.0
            output_tot[0,5] = out[i,ix,-1,3]+(out[i,ix,-1,3]-out[i,ix,-2,3])/2.0
            output_tot[0,6] = out[i,ix,-1,4]+(out[i,ix,-1,4]-out[i,ix,-2,4])/2.0
            output_tot[0,7] = out[i,ix,-1,5]+(out[i,ix,-1,5]-out[i,ix,-2,5])/2.0
            np.savetxt(f, output_tot, fmt='%g')
        for iy in range(0,ny):
            y = L_y/(ny*2.0)+iy*(L_y/ny)
            #output_tot = np.zeros((1, 8))
            output_tot[0,0] = 0
            output_tot[0,1] = y
            output_tot[0,2] = out[i,0,iy,0]-(out[i,1,iy,0]-out[i,0,iy,0])/2.0
            output_tot[0,3] = out[i,0,iy,1]-(out[i,1,iy,1]-out[i,0,iy,1])/2.0
            output_tot[0,4] = out[i,0,iy,2]-(out[i,1,iy,2]-out[i,0,iy,2])/2.0
            output_tot[0,5] = out[i,0,iy,3]-(out[i,1,iy,3]-out[i,0,iy,3])/2.0
            output_tot[0,6] = out[i,0,iy,4]-(out[i,1,iy,4]-out[i,0,iy,4])/2.0
            output_tot[0,7] = out[i,0,iy,5]-(out[i,1,iy,5]-out[i,0,iy,5])/2.0
            np.savetxt(f, output_tot, fmt='%g')
            output_tot[0,0] = L_x
            #output_tot[0,1] = y
            output_tot[0,2] = out[i,-1,iy,0]+(out[i,-1,iy,0]-out[i,-2,iy,0])/2.0
            output_tot[0,3] = out[i,-1,iy,1]+(out[i,-1,iy,1]-out[i,-2,iy,1])/2.0
            output_tot[0,4] = out[i,-1,iy,2]+(out[i,-1,iy,2]-out[i,-2,iy,2])/2.0
            output_tot[0,5] = out[i,-1,iy,3]+(out[i,-1,iy,3]-out[i,-2,iy,3])/2.0
            output_tot[0,6] = out[i,-1,iy,4]+(out[i,-1,iy,4]-out[i,-2,iy,4])/2.0
            output_tot[0,7] = out[i,-1,iy,5]+(out[i,-1,iy,5]-out[i,-2,iy,5])/2.0
            np.savetxt(f, output_tot, fmt='%g')



# not do velocity reconstruction for 3D
#print("Finished field output. Beginning velocity-field reconstruction ... ...")

#num_layers = 25 # number of sigma-layers in each layer
#for i in range(0, frames):
    #print('Reconstructing'+': '+str(i)+'th output')
    #t = 1.0*i
    #format_string_time = f"{t:.1f}"
    #file_name = 'reconstructed_%s' % format_string_time
    #with open(file_name, 'w') as f:
        #writer = csv.writer(f, delimiter='\t')
        #for cell in range(0,arr_size):
            #x = L_x/(nx*2.0)+cell*(L_x/nx)
            #hl = out[i,cell,0]
            #hu = out[i,cell,1]
            #ul = out[i,cell,2]
            #uu = out[i,cell,3]
            #c11 = (-6.0*hu*ul+9.0*hl*(-2.0*ul+uu)*μr_v)/(hl**2.0*(4.0*hu+3.0*hl*μr_v))
            #c12 = (12.0*hu*ul+18.0*hl*ul*μr_v-6.0*hl*uu*μr_v)/(hl*(4.0*hu+3.0*hl*μr_v))
            #c21 = (9.0*ul-6.0*uu)/(hu*(4.0*hu+3.0*hl*μr_v))
            #c22 = (6.0*(hl+hu)*(2.0*uu-3.0*ul))/(hu*(4.0*hu+3.0*hl*μr_v))
            #c23 = (3.0*(2.0*hu**2.0*ul+hl**2.0*(3.0*ul-2.0*uu)+hl*hu*(6.0*ul+uu*(-4.0+μr_v))))/(hu*(4.0*hu+3.0*hl*μr_v))
            ## accomodate the boundary conditions at the wall
            #output_tot = np.zeros((1, 3))
            #output_tot[0,0] = x
            #np.savetxt(f, output_tot, fmt='%g')
            ## begin with lower layer
            #for layer in range(0,num_layers):
                #output_tot = np.zeros((1, 3))
                #z = hl/(num_layers*2.0)+layer*(hl/num_layers)

                #u = c11*z*z+c12*z
                #output_tot[0,0] = x
                #output_tot[0,1] = z
                #output_tot[0,2] = u
                #np.savetxt(f, output_tot, fmt='%g')
                #if cell == 0:
                    #output_tot[0,0] = 0.0
                    #np.savetxt(f, output_tot, fmt='%g')
                #if cell == (arr_size-1):
                    #output_tot[0,0] = L_x
                    #np.savetxt(f, output_tot, fmt='%g')
            ## the upper layer
            #for layer in range(0,num_layers):
                #output_tot = np.zeros((1, 3))
                #z = hl+hu/(num_layers*2.0)+layer*(hu/num_layers)
                #u = c21*z*z+c22*z+c23
                #output_tot[0,0] = x
                #output_tot[0,1] = z
                #output_tot[0,2] = u
                #np.savetxt(f, output_tot, fmt='%g')
                #if cell == 0:
                    #output_tot[0,0] = 0.0
                    #np.savetxt(f, output_tot, fmt='%g')
                #if cell == (arr_size-1):
                    #output_tot[0,0] = L_x
                    #np.savetxt(f, output_tot, fmt='%g')
            ## make sure the contour hits the free-surface
            #z = hl+hu
            #u = c21*z*z+c22*z+c23
            #output_tot[0,0] = x
            #output_tot[0,1] = z
            #output_tot[0,2] = u
            #np.savetxt(f, output_tot, fmt='%g')
            #if cell == 0:
                    #output_tot[0,0] = 0.0
                    #np.savetxt(f, output_tot, fmt='%g')
            #if cell == (arr_size-1):
                #output_tot[0,0] = L_x
                #np.savetxt(f, output_tot, fmt='%g')


