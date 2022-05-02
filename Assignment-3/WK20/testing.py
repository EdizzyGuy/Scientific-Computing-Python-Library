import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_heat_eq import *
from utility import *
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import time

L = 1.0
T = 0.5
kappa = 1.0

mx = 100
print(mx)
mt = find_suitable_mt(mx, kappa, L, T)
print(mt)
lmbda = kappa*T*mx**2 / (mt*L**2)
print(lmbda)

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
count = 2
avg, std = time_function(forw_eul_heat_eq, count, args=args, kwargs=kwargs)
print(f'avg time : {avg} \n Standard deviation in {count} calculations : {std}')

sol = forw_eul_heat_eq(t,x,u_I,kappa,method='Singular')
u_j = sol[-1]

from utility import mean_square_error
anal_sol = u_exact(x, T)
mse = mean_square_error(u_j, u_exact)
print(mse)

# Plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

