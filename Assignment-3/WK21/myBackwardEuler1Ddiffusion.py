#%%
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import scipy
from scipy import linalg
import time

# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=1.0         # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

#%%
# Set numerical parameters
mx = 100     # number of gridpoints in space
mt = 10000   # number of gridpoints in time

# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)

#%%
# Set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jp1 = np.zeros(x.size)      # u at next time step

# INITIAL IMPLEMENTATION:
# Set initial condition
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])

for i in range(0, mx+1):
    u_j[i] = u_I(x[i])
be_start = time.time()
u_j_BE = u_j
lambda_array = np.array([lmbda for i in range(mx - 1)])
A_BE_1 = np.diag(1 + 2* lambda_array)
A_BE_2 = np.diag(-1 * lambda_array[:-1], k=1)
A_BE_3 = np.diag(-1 * lambda_array[:-1], k=-1)
A_BE = A_BE_1 + A_BE_2 + A_BE_3
# MATRIX be MAY CHANGE EACH TIME STEP
# A_BE * u_j+1 = u_j
# u_j+1 = inv(A_BE) * u_j
inv_A_BE = np.linalg.inv(A_BE)
for j in range(0, mt):
    u_j_BE[1:-1] = np.matmul(inv_A_BE, u_j_BE[1:-1])
    u_j_BE[[0, -1]] = 0
be_finish = time.time()
be_duration = be_finish - be_start

#%%
# Plot the final result and exact solution
plot_graph = True
if plot_graph is True:
    pl.plot(x,u_j_BE,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()

BE_mse = 0
for i in range(len(u_j_BE)):
    BE_mse += (u_j_BE[i] - u_exact(xx, T)[i])**2
BE_mse /= len(u_j_BE)

print('time elapsed :', be_duration)
print('mean square error of numerical and analytical solution :', BE_mse)


# %%
# test functional implementation in solve_heat_eq.py
from solve_heat_eq2 import *

u = back_eul_heat_eq(t, x, u_I, kappa)
sol = u[-1]
pl.plot(x,sol,'ro',label='from function')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

# %%
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *
mse = mean_square_error(sol, u_j_BE)
print('mean square error between scripts solution'\
        f' and functions solution is {mse:.2E}')

# %%
