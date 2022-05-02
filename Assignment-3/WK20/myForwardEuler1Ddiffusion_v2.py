# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# %%
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import time

# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5         # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 31     # number of gridpoints in space
mt = 1000   # number of gridpoints in time

# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)
# MAYBE MAKE IT INTO AN ERROR
if lmbda > 0.5:
    print('Euler stability criterion not met, solutions will be incorrect!')

# Set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jp1 = np.zeros(x.size)      # u at next time step

# INITIAL IMPLEMENTATION:
# Set initial condition
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])

method = 'Matrix'
fe_start = time.time()
match method:
    case 'singular':
        # Solve the PDE: loop over all time points
        for j in range(0, mt):
            # Forward Euler timestep at inner mesh points
            # PDE discretised at position x[i], time t[j]
            for i in range(1, mx):
                u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])  # find solution forward 1 step in time
                
            # Boundary conditions
            u_jp1[0] = 0; u_jp1[mx] = 0
                
            # Save u_j at time t[j+1]
            u_j[:] = u_jp1[:]

    case 'Matrix':
        # MATRIX IMPLEMENTATION
        fe_start = time.time()  # match statements not great for time
        # ^WRITE ABOUT THIS
        lambda_array = np.array([lmbda for i in range(mx - 1)])
        A_FE_1 = np.diag(1 - 2* lambda_array)
        A_FE_2 = np.diag(lambda_array[:-1], k=1)
        A_FE_3 = np.diag(lambda_array[:-1], k=-1)
        A_FE = A_FE_1 + A_FE_2 + A_FE_3

        for j in range(0, mt):
            u_j[1:-1] = np.matmul(A_FE, u_j[1:-1])
            u_j[[0,-1]] = 0
fe_end = time.time()
fe_duration = fe_end - fe_start

FE_mse = 0
for i in range(len(u_j)):
    FE_mse += (u_j[i] - u_exact(xx, T)[i])**2
FE_mse /= len(u_j)

# Plot the final result and exact solution
plot_graph = False
if plot_graph is True:
    pl.plot(x,u_j,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


print('time elapsed :', fe_duration)
print('mean square error of numerical and analytical solution :', FE_mse)
