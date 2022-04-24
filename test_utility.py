import numpy as np
from utility import get_phase_portrait
from scipy.integrate import solve_ivp

import unittest

def dX_dt(t, X):
    X_dot = np.array([X[1], -X[0]])
    return X_dot


class TestStringMethods(unittest.TestCase):

# TEST FOR HIGHER DIMENSIONAL SYS OF ODES
    def test_get_phase_portrait_mySolver(self):
        # known period of 2*pi THIS IS KINDA TESTING SOLVER AGAIN
        init_cond = np.array([0, 1])
        period = 2 *np.pi
        path = get_phase_portrait(dX_dt, init_cond, solve_for=(0, period))

        self.assertTrue(np.all(np.isclose(path[0], path[-1])))

    def test_get_phase_portrait_scipySolver(self):
        # known period of 2*pi THIS IS KINDA TESTING SOLVER AGAIN
        init_cond = np.array([0, 1])
        period = 2 *np.pi
        path = get_phase_portrait(dX_dt, init_cond, solve_for=(0, period), solver=solve_ivp)

        self.assertTrue(np.all(np.isclose(path[0], path[-1])))

if __name__ == '__main__':
    unittest.main()