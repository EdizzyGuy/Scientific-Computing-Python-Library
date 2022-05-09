import sys
import os
path = sys.path[0]
origin = os.path.dirname(path)
sys.path.append(origin)

import unittest
import numpy as np
from ode import solve_ode

def dx_dt(t, x):
    x_dot = x
    return x_dot

def dX_dt(t, X):
    X_dot = np.array([X[1], -X[0]])
    return X_dot

def test_solve_ode_1D_RK45():
    T = 42
    # for x_dot = x and x0 = 1, analytical equation is x(t) = e^t
    anal_eq = lambda t : np.exp(t)
    anal_sol = anal_eq(T)

    x0 = np.array([1])
    path = solve_ode(dx_dt, x0, solve_for=[0, T], deltat_max=1e-03, method='RK45')
    numer_sol = path[-1]

    return np.isclose(numer_sol, anal_sol, rtol=1e-02)

test_solve_ode_1D_RK45()
