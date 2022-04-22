import sys
import os
path = sys.path[0]
parent = os.path.dirname(path)
sys.path.append(parent)


import numpy as np
import ode

""" x_ddot = -x
therefore system of ODE to solve is :
X = [x, x_dot]
X_dot = [x_dot, -x]

let initial conditions be   x(0)        = 0
and                         x_dot(0)    = 1
therefore x should solve to sin(t)"""


def dX_dt(t, X):
    X_dot = np.array([X[1], -X[0]])
    return X_dot


initial_condition = np.array([0, 1])
solve_for = np.linspace(0, 2*np.pi, 6)
time_step = 0.0001

result_euler = ode.solve_ode(dX_dt, initial_condition, solve_for, deltat_max=time_step, method='Euler')
result_rk4 = ode.solve_ode(dX_dt, initial_condition, solve_for, deltat_max=time_step, method='RK4')

true_results = np.array([np.sin(solve_for), np.cos(solve_for)]).transpose()
mse_euler = np.square(result_euler - true_results).mean()
mse_rk4 = np.square(result_rk4 - true_results).mean()

print(f'mean square error of Euler method is        : {mse_euler:.2}')
print(f'mean square error of Runge-Kutta method is  : {mse_rk4:.2}')

