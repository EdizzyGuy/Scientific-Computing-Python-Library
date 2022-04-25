'''solve_ode can solve time variant systems of ode's although it should be noted that init time is implicitly defined as solve_for[0]
ADD FUNCTIONALITY TO SPECIFY INITIAL TIME THAT IS DIFFERENT THAN SOLVE_FOR[0]
'''

import sys
import os
path = sys.path[0]
origin = os.path.dirname(path)
sys.path.append(origin)

import numpy as np
from ode import solve_ode
# solving t^2 *y'' + 4ty' + 2y = 0
# with init condition y(1) = 1, & y'(1) = 1
# will give analytical sol y(t) = 3*t^-1 - 2*t^-2
#                       and y'(t) = -3*t^-2 + 4 t^-3

def analytical_Y(t):
    y = 3/t - 2/t**2
    y_dot = -3/t**2 + 4/t**3
    return np.array([y, y_dot])

def analytical_Ydot(t):
    y_dot = -3/t**2 + 4/t**3
    y_ddot = 6/t**3 - 12/t**4
    return np.array([y_dot, y_ddot])

def Euler_Cauchy(t, Y):
    y = Y[0]
    y_dot = Y[1]
    y_ddot = -4*y_dot/t - 2*y/t**2
    return np.array([y_dot, y_ddot])


init_cond = np.array([1,1])
init_time = 1
T = 10

assert np.all(np.isclose(analytical_Ydot(init_time), Euler_Cauchy(init_time, init_cond)))


solve_for = [init_time, T]
results = solve_ode(Euler_Cauchy, init_cond, solve_for)
numer_sol = results[-1]
anal_sol = analytical_Y(T)

assert np.all(np.isclose(anal_sol, numer_sol))

