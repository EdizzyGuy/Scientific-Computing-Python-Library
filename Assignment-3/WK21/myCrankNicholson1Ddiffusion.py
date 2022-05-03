#%%
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import scipy
import time

'''
This script implements crank nicholson on the diffusion equation and grabs useful metrics.
Also uses crank nicholson function in solve_heat_eq2 to do the same

--------------------------------------------------------------------------------
Average time of crank nicholson using matrix inversion : 2.22
Standard deviation                                     : 1.09E-01

Average time of crank nicholson using matrix inversion : 2.22
Standard deviation                                     : 1.09E-01
'''
#%%
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
mx = 100     # number of gridpoints in space
#INCREASING MX FROM 20 TO 40 INCREASED ACCURACY
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

anal_sol = u_exact(x,T)

# Set up the solution variables
u_j_CN = np.zeros(x.size)        # u at current time step
u_jp1_CN = np.zeros(x.size)      # u at next time step

#%%
from scipy import linalg

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
    u_jp1_CN = linalg.solve(A_CN, b)
    u_j_CN[1: -1] = u_jp1_CN  
cn_finish = time.time()
cn_duration = cn_finish - cn_start
print(f'duration : {cn_duration}')

# REMEMBER PROBLEMS WITH MATRIX INVERSION

#%%
# Plot the final result and exact solution
plot_graph = True
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

# %%
#see if function in solve_heat_eq2 works
from solve_heat_eq2 import *
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

u = crank_nichol_eul_heat_eq(t, x, u_I, kappa, method='linalg solve', u_I_args=tuple())
sol = u[-1]

pl.plot(x,u[-1],'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,1)')
pl.legend(loc='upper right')
pl.show()

mse_ls = mean_square_error(u[-1], anal_sol)
print(f'mean square error between analytical and numerical LINALG.SOLVE {mse_ls}')

#%%

u_matinv = crank_nichol_eul_heat_eq(t, x, u_I, kappa, method='matrix inversion', u_I_args=tuple())
mse_mi = mean_square_error(u_matinv[-1], anal_sol)
print(f'mean square error between analytical and numerical MATRIX_INVERSION {mse_mi}')


pl.plot(x,u_matinv[-1],'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,1)')
pl.legend(loc='upper right')
pl.show()

print(f'mse of matrix inversion / mse of linalg solve {mse_mi/mse_ls:.2f}')
# %%

# TODO DEFINE ANALYTICAL SOLUTION AT THE CORRECT TIMESTEPS AND COMPARE
#EACH METHODS MSE AND COMPLETION TIME

anal_sol = u_exact(x,T)
#crank_nichol_eul_heat_eq(t, x, u_I, kappa, method='matrix inversion', u_I_args=tuple())
args = (t,x,u_I,kappa)
kwargs_inv = {'method' : 'matrix inversion'}
kwargs_linalg = {'method' : 'linalg solve'}

inv_avg, inv_std = time_function(crank_nichol_eul_heat_eq, 10, args, kwargs_inv)
linalg_avg, linalg_std = time_function(crank_nichol_eul_heat_eq, 10, args, kwargs_linalg)


#%%
print(f'Average time of crank nicholson using matrix inversion : {inv_avg:.2f}\n'
      f'Standard deviation                                     : {inv_std:.2E}\n')
      
print(f'Average time of crank nicholson using linalg.solve     : {linalg_avg:.2f}\n'
      f'Standard deviation                                     : {linalg_std:.2E}\n')

print(f'Time of script : {cn_duration:.2f}')
# %%

