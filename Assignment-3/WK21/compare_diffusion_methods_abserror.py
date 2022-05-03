#%%
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

from solve_heat_eq2 import *
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import scipy
import time


#%%
# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.2         # total time to solve for
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
#mt = find_suitable_mt(mx, kappa, L, T)   # number of gridpoints in time
mt = 100

# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)


#%% COMPARE METHODS
anal_u = u_exact(x,T)
args = (t,x,u_I,kappa)
kwargs_fe = {'method' : 'Matrix'}
kwargs_cn = {'method' : 'matrix inversion'}

u_fe = forw_eul_heat_eq(t,x,u_I,kappa,**kwargs_fe)
u_be = back_eul_heat_eq(t, x, u_I, kappa)
u_cn = crank_nichol_eul_heat_eq(t, x, u_I, kappa, **kwargs_cn, u_I_args=tuple())

fe_mse = abs_error(u_fe[-1], anal_u)
be_mse = abs_error(u_be[-1], anal_u)
cn_mse = abs_error(u_cn[-1], anal_u)


count = 1000
fe_avg, fe_std = time_function(forw_eul_heat_eq, count, args, kwargs_fe)
be_avg, be_std = time_function(back_eul_heat_eq, count, args)
cn_avg, cn_std = time_function(crank_nichol_eul_heat_eq, count, args, kwargs_cn)


#%% PRINT METRICS

print('FORWARD EULER METRICS')
print('---------------------')
print(f'Sum of abs error between numerical and analytical solution  : {fe_mse:.2E}')
print(f'Average duration, in {count} runs, to complete task\t     : {fe_avg:.2E}')
print(f'Standard deviation of duration to complete task in {count} runs : {fe_std:.2E}')
print()

print('BACKWARD EULER METRICS')
print('---------------------')
print(f'Sum of abs error between numerical and analytical solution : {be_mse:.2E}')
print(f'Average duration to complete task in {count} runs\t\t    : {be_avg:.2E}')
print(f'Standard deviation of duration to complete task in {count} runs : {be_std:.2E}')
print()

print('CRANK-NICHOLSON METRICS')
print('----------------------')
print(f'Sum of abs error between numerical and analytical solution  : {cn_mse:.2E}')
print(f'Average duration to complete task in {count} runs\t\t     : {cn_avg:.2E}')
print(f'Standard deviation of duration to complete task in {count} runs : {cn_std:.2E}')
print()


# %%
# COMPARE RUN TIMES OF EACH FOR SIMILAR MEAN SQ ERRORS
# COMOARE RUN TIMES OF FE AND BE FOR SIMILAR MEAN SQ ERRORS

mx_be, mt_be = mx, mt
mse_close = np.isclose(be_mse, fe_mse, atol=0, rtol=0.05)
while not mse_close and mt_be > 1:
    mt_be -= 1

    t_be, x_be = get_grid_space(T, L, mt_be, mx_be)
    u_be = back_eul_heat_eq(t_be, x_be, u_I, kappa)
    be_mse_vary_mt = abs_error(anal_u, u_be[-1])

    mse_close = np.isclose(be_mse_vary_mt, fe_mse, atol=0, rtol=0.05)
    #print(mse_close)

print(f'Sum of abs errors of forward euler and backward euler within tolerance : \n'
        f'{mse_close}')


#%%
# since fe mse is largest use that as a baseline
mx_cn, mt_cn = mx, mt
mse_close = np.isclose(cn_mse, fe_mse, atol=0, rtol=0.05)
while not mse_close and mt_cn > 1:
    mt_cn -= 1

    t_cn, x_cn = get_grid_space(T, L, mt_cn, mx_cn)
    u_cn = crank_nichol_eul_heat_eq(t_cn, x_cn, u_I, kappa, **kwargs_cn, u_I_args=tuple())
    cn_mse_vary_mt = abs_error(anal_u, u_cn[-1])

    mse_close = np.isclose(cn_mse_vary_mt, fe_mse, atol=0, rtol=0.05)
    #print(mse_close)

print(f'Sum of abs errors of forward euler and crank nicholson within tolerance : \n'
        f'{mse_close}')



#%%
args = (t_be,x_be,u_I,kappa)
be_avg_dmt, be_std_dmt = time_function(back_eul_heat_eq, count, args)

print(f'For sum of abs error within 5% of each other between backward euler'
        'and forward euler:\n')
print(f'Forward euler average time   : {fe_avg:.2E}')
print(f'backward euler average time : {be_avg_dmt:.2E}')


#%%
args = (t_cn,x_cn,u_I,kappa)
cn_avg_dmt, cn_std_dmt = time_function(crank_nichol_eul_heat_eq, count, args, kwargs_cn)

print(f'For sum of abs error within 5% of each other between Crank Nicholson'
        'and forward euler:\n')
print(f'Forward euler average time   : {fe_avg:.2E}')
print(f'Crank Nicholson average time : {cn_avg_dmt:.2E}')


# %%
