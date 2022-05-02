import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_heat_eq import *
from utility import *
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import time

'''
This script showcases the validity of the forward euler stepwise discretization method defined in solve_heat_eq.py
and also defined the draft for an animation function that showcases the solution forwards in time.

Calculating average time for forward euler, singular stepwise implementation
Average time in 10 calculations         : 1.81E+00s 
Standard deviation in 10 calculations   : 1.79E-02s

Mean square error of forward euler in for solving the heat equation is 1.69E-11
'''

L = 1.0
T = 0.5
kappa = 1.0
print(f'Solving the heat equation for domain of length {L}, for time {T} with kappa {kappa}')
print(f'Solving using discretized forward Euler')

mx = 100
print(f'Allocated {mx}\tgridpoints in space')
mt = find_suitable_mt(mx, kappa, L, T)
print(f'Allocated {mt}\tgridpoints in time')
print()

lmbda = kappa*T*mx**2 / (mt*L**2)
print(f'Value of lambda : {lmbda}')
print()


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

t, x = get_grid_space(T,L,mt,mx)
args=(t,x,u_I,kappa)
kwargs= {'method' : 'Singular'}
count = 1
print(f'Calculating average time for forward euler, singular stepwise implementation')
avg, std = time_function(forw_eul_heat_eq, count, args=args, kwargs=kwargs)
print(f'Average time in {count} calculations\t\t: {avg:.2E}s \nStandard deviation in {count} calculations\t: {std:.2E}s')
print()

sol = forw_eul_heat_eq(t,x,u_I,kappa,method='Singular')
u_j = sol[-1]

from utility import mean_square_error
anal_sol = u_exact(x, T)
mse = mean_square_error(u_j, anal_sol)
print(f'Mean square error of forward euler in for solving the heat equation is {mse:.2E}')
print()

# Plot the final result and exact solution
pl.plot(x,u_j,'ro',label='numerical')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='analytical')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

