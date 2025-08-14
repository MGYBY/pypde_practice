import matplotlib.pyplot as plt
import numpy as np

from hyperbopy import Simulation
from hyperbopy.models import SW2LLaminar_simp

# initial conditions
def disturbed_depth(x, normal_depth, wl):
    return (normal_depth*(1.0+dist_amp*np.sin(np.pi*2.0*x/wl)))

def u2_su(hr, ρr, μr):
    return ((3.0*μr+2.0*hr*ρr*(hr+3.0*μr))/(μr*(2.0+3.0*hr*ρr)))

# def bss(hl, hu, ul, uu, μr):
#     return ((12.0*hu*ul+18.0*hl*ul*μr-6.0*hl*uu*μr)/(4.0*hl*hu+3.0*hl**2.0*μr))

# ## Grid parameters
tmax = 16.0
Nx = 150
L_x = 1.0
dist_amp = 0.1
x = np.linspace(0, L_x, Nx)
dx = np.diff(x)[0]

# problem-related parameters
fr_v_v = 0.80
hr_v_v = 1.0
μr_v_v = 1.0
ρr_v_v = 1.0

nd1 = 1.0
nd2 = nd1*hr_v_v
nv1 = 1.0
nv2 = u2_su(hr_v_v, ρr_v_v, μr_v_v)

# initial conditions
h1 = np.empty_like(x)
h1 = disturbed_depth(x, nd1, L_x)

h2 = np.empty_like(x)
h2 = disturbed_depth(x, nd2, L_x)

# q1, q2 = np.zeros_like(x), np.zeros_like(x)
q1 = nv1*h1
q2 = nv2*h2

Z = 0.0+np.zeros_like(x)

W0 = np.array([h1, q1, h2, q2, Z])

# ## Boundary conditions
BCs = [['periodic', 'periodic'], ['periodic', 'periodic'],
       ['periodic', 'periodic'], ['periodic', 'periodic']]

# ## Initialization
model = SW2LLaminar_simp(fr_v=fr_v_v, ρr_v=ρr_v_v, μr_v=μr_v_v, hr_v=hr_v_v)  # model with default parameters
simu = Simulation(
    model, W0, BCs, dx, temporal_scheme='RungeKutta33', spatial_scheme='CentralUpwindPathConservative', dt_fact=0.1)  # simulation
    # model, W0, BCs, dx, temporal_scheme='RungeKutta33', spatial_scheme='CentralUpwind', dt_fact=0.1)  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=25, x=x, Z=Z)

# %% plot final figure
fig, ax = plt.subplots(1, 1, layout='constrained')

color = 'tab:blue'
ax.plot(x, Z + U[-1, 0, :] + U[-1, 2, :], color=color)
ax.set_ylabel('water surface [m]', color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.plot(x, Z + U[-1, 2, :], color=color)
ax2.set_ylabel('interface [m]', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax.set_xlabel('x [m]')

np.savetxt('PCCU_simp.txt', np.c_[x,U[-1, 0, :], U[-1, 2, :]])

plt.show()
