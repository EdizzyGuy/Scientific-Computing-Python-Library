#%%
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import scipy
import time
#%%
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

# Set numerical parameters
mx = 10     # number of gridpoints in space
#INCREASING MX FROM 20 TO 40 INCREASED ACCURACY
mt = 100   # number of gridpoints in time

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

# Set up the solution variables
u_j_CN = np.zeros(x.size)        # u at current time step
u_jp1_CN = np.zeros(x.size)      # u at next time step

# INITIAL IMPLEMENTATION:
# Set initial condition
for i in range(0, mx+1):
    u_j_CN[i] = u_I(x[i])

cn_start = time.time()
lambda_array = np.array([lmbda for i in range(mx - 1)])
# A_CN = tridiag(-lmbda/2, 1+lambda, -lambda/2)
A_CN_1 = np.diag(1 + lambda_array)
A_CN_2 = np.diag(-1/2 * lambda_array[:-1], k=1)
A_CN_3 = np.diag(-1/2 * lambda_array[:-1], k=-1)
A_CN = A_CN_1 + A_CN_2 + A_CN_3
# B_CN = tridiag(lamdba/2, 1-lamdba, lamdba/2)
B_CN_1 = np.diag(1 - lambda_array)
B_CN_2 = np.diag(1/2 * lambda_array[:-1], k=1)
B_CN_3 = np.diag(1/2 * lambda_array[:-1], k=-1)  # CAN JUST USE MATRIX TRANSPOSE OF B_CN_2
B_CN = B_CN_1 + B_CN_2 + B_CN_3
# A_CN * u_j+1 = B_CN * u_j
# u_j+1 = inv(A_CN) * B_CN * u_j
for j in range(0, mt):
    # test = u_j_CN[1:-1]
    # u_j_CN[1:-1] = inv_A_CN @ B_CN @ test
    b = B_CN @ u_j_CN[1:-1]
    u_jp1_CN = scipy.linalg.solve(A_CN, b)
    u_j_CN[1: -1] = u_jp1_CN  
cn_finish = time.time()
cn_duration = cn_finish - cn_start

# REMEMBER PROBLEMS WITH MATRIX INVERSION

# Plot the final result and exact solution
plot_graph = False
if plot_graph is True:
    pl.plot(x,u_j_CN,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()

CN_mse = 0
for i in range(len(u_j_CN)):
    CN_mse += (u_j_CN[i] - u_exact(xx,T)[i])**2
CN_mse /= len(u_j_CN)

print('time elapsed :', cn_duration)
print('mean square error of numerical and analytical solution :', CN_mse)
