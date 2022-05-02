import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_heat_eq import *
from utility import *
import numpy as np
from math import pi
import time

'''
This python file explores how changing the power of the sinusoid in the initial condition can change solutions

'''
# initialise problem
L = 1.0
T = 0.2
kappa = 1.0

mx = 100
mt = 10000
lmbda = kappa*T*mx**2 / (mt*L**2)

def u_I(x, power):
    # initial temperature distribution
    y = pow(np.sin(pi*x/L), power)
    return y

t, x = get_grid_space(T,L,mt,mx)

# want animation to last 10 sec therefore find suitable fps


# iterate over different powers and showcase solutions
powers = [i for i in range(1, 5)]
for power in powers:
    args= (power,)
    title = r'Initial condition : $\sin^' f'{power}' r'(\pi x)$'

    u = forw_eul_heat_eq(t,x,u_I,kappa,method='Matrix', u_I_args=args)
    # since mt is too high, i will artificially reduce the frames encompassed within u
    # also discarding solutions past t = 0.25 because they are boring
    u_anim = u[0:int(mt/2):40]
    t_anim = t[0:int(mt/2):40]
    fps = len(t_anim) / 10
    display_dynamic_solution(u_anim, x, t_anim, title, fps)

