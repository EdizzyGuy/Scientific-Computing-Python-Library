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


class TestStringMethods(unittest.TestCase):

    def test_solve_ode_1D_Euler(self):
        T = 42
        # for x_dot = x and x0 = 1, analytical equation is x(t) = e^t
        anal_eq = lambda t : np.exp(t)
        anal_sol = anal_eq(T)

        x0 = np.array([1])
        path = solve_ode(dx_dt, x0, solve_for=[0, T], deltat_max=1e-04, method=ode.euler_step)
        numer_sol = path[-1]

        self.assertTrue(np.isclose(numer_sol, anal_sol, rtol=1e-02))

    def test_solve_ode_1D_RK4(self):
        T = 42
        # for x_dot = x and x0 = 1, analytical equation is x(t) = e^t
        anal_eq = lambda t : np.exp(t)
        anal_sol = anal_eq(T)

        x0 = np.array([1])
        path = solve_ode(dx_dt, x0, solve_for=[0, T])
        numer_sol = path[-1]

        self.assertTrue(np.isclose(numer_sol, anal_sol))

    def test_solve_ode_1D_RK45(self):
        T = 42
        # for x_dot = x and x0 = 1, analytical equation is x(t) = e^t
        anal_eq = lambda t : np.exp(t)
        anal_sol = anal_eq(T)

        x0 = np.array([1])
        path = solve_ode(dx_dt, x0, solve_for=[0, T], deltat_max=1e-03, method='RK45')
        numer_sol = path[-1]

        self.assertTrue(np.isclose(numer_sol, anal_sol, rtol=1e-02))

    def test_solve_ode_sys_Euler(self):
        T = 7
        init_cond = np.array([0,1])
        # for sys of ode and init cond, analytical sol is x = sin(t) and x_dot = cos(t)
        anal_eq = lambda t : np.array([np.sin(t), np.cos(t)])
        anal_sol = anal_eq(T)

        path = solve_ode(dX_dt, init_cond, solve_for=[0, T], deltat_max=1e-04, method=ode.euler_step)
        numer_sol = path[-1]
        self.assertTrue(np.all(np.isclose(numer_sol, anal_sol, rtol=1e-02)))

    def test_solve_ode_sys_RK4(self):
        T = 7
        init_cond = np.array([0,1])
        # for sys of ode and init cond, analytical sol is x = sin(t) and x_dot = cos(t)
        anal_eq = lambda t : np.array([np.sin(t), np.cos(t)])
        anal_sol = anal_eq(T)

        path = solve_ode(dX_dt, init_cond, solve_for=[0, T])
        numer_sol = path[-1]
        self.assertTrue(np.all(np.isclose(numer_sol, anal_sol)))
        

if __name__ == '__main__':
    unittest.main()