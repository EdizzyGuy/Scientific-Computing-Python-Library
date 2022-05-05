import numpy as np
from utility import *
from scipy.integrate import solve_ivp

import unittest

''' run by passing python -m unittest -v test_utility to cmd 
TEST FUNCTIONS MUST START WITH test'''

def dX_dt(t, X):
    X_dot = np.array([X[1], -X[0]])
    return X_dot


class TestStringMethods(unittest.TestCase):

# TEST FOR HIGHER DIMENSIONAL SYS OF ODES
    def test_get_phase_portrait_mySolver(self):
        # known period of 2*pi THIS IS KINDA TESTING SOLVER AGAIN
        atol = 1e-02
        init_cond = np.array([0, 1])
        period = 2 *np.pi
        path = get_phase_portrait(dX_dt, init_cond, solve_for=(0, period))

        self.assertTrue(np.all(np.isclose(path[0], path[-1], atol=atol)))

    def test_get_phase_portrait_scipySolver(self):
        # known period of 2*pi THIS IS KINDA TESTING SOLVER AGAIN
        atol = 1e-02

        init_cond = np.array([0, 1])
        period = 2 *np.pi
        path = get_phase_portrait(dX_dt, init_cond, solve_for=(0, period), solver=solve_ivp)

        self.assertTrue(np.all(np.isclose(path[0], path[-1], atol=atol)))

    def test_abs_error(self):
        a = np.arange(10)
        b = 10*a

        true_abs_error = 45*9
        abs_err = abs_error(a,b)

        self.assertTrue(true_abs_error == abs_err)

    def test_mse(self):
        a = np.arange(10)
        b = 10*a

        true_mse = 81*285 / 10
        mse = mean_square_error(a,b)

        self.assertTrue(true_mse == mse)

    def test_rel_error(self):
        a = np.arange(10)
        b = 10*a

        true_rel_err = 10
        rel_err = mean_rel_error(b,a)

        self.assertTrue(true_rel_err == rel_err)

    def test_isInteger(self):
        a = 1 + 1e-10
        b = 0.2

        true_ans = (True, False)
        ans = (isInteger(a), isInteger(b))

        self.assertTrue(ans == true_ans)
        


if __name__ == '__main__':
    unittest.main()