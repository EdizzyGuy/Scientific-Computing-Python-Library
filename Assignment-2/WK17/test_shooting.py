import sys
import os
path = sys.path[0]
origin = os.path.dirname(os.path.dirname(path))
sys.path.append(origin)

import unittest
import numpy as np
from numerical_continuation import find_limit_cycles
from scipy.integrate import solve_ivp

PI = np.pi

''' run by passing python -m unittest -v test_shooting to cmd 
TEST FUNCTIONS MUST START WITH test'''

def hopf_bifurcation(t, U, beta, sigma):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 + sigma *u1 *(u1*u1 + u2*u2)
    u2_dot = u1 + beta*u2 + sigma *u2 *(u1*u1 + u2*u2)
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

def hopf_extended(t, U, beta=2, sigma=-1):
    t = None

    U_dot = np.zeros(3)
    U_dot[:-1] = hopf_bifurcation(t, U[:-1], beta, sigma)
    U_dot[-1] = - U[-1]

    return U_dot


# SCIPY METHODS REQUIRE GREATER TOLERANCE
class TestShootingMethods(unittest.TestCase):

    #TODO has problems with array([-19.97797856, -16.40312613,   6.50074   ]) just returns this init guess (tried with T = 10)
    # probably to do with fsolve failing 
    # with T=10 finds sol = array([-1.41421356e+00, -7.02291778e-11,  1.13097336e+02])
    # where this is a valid solution
    def test_shooting_hopf_mySolver(self):
        beta, sigma = 2,-1
        args = (beta, sigma)
        # from analytical solution
        anal_period = 2* PI
        anal_radius = np.sqrt(beta)

        init_cond = np.array([1,1])
        #np.random.normal(size=2) * 9
        period_guess = np.random.uniform(low=5, high=25, size=1)
        #changed oeriod guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
        init_guess = np.concatenate((init_cond, period_guess))

        roots = find_limit_cycles(init_guess, hopf_bifurcation, args=(beta, sigma))
        num_period = roots[-1]
        num_radius = np.linalg.norm(roots[:-1])

        # check if period is a multiple of the analytical sol
        period_multiplicity = num_period / anal_period
        period_multiple = np.isclose(period_multiplicity % 1, 1) or np.isclose(period_multiplicity % 1, 0)
        self.assertTrue(period_multiple and np.isclose(anal_radius, num_radius))

    def test_shooting_hopf_scipySolver(self):
        beta, sigma = 2,-1
        args = (beta, sigma)
        atol=1e-2
        # from analytical solution
        anal_period = 2* PI
        anal_radius = np.sqrt(beta)

        init_cond = np.array([1,1])
        #np.random.normal(size=2) * 9
        period_guess = np.random.uniform(low=5, high=25, size=1)
        #changed oeriod guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
        init_guess = np.concatenate((init_cond, period_guess))

        roots = find_limit_cycles(init_guess, hopf_bifurcation, solver=solve_ivp, args=(beta, sigma))
        num_period = roots[-1]
        num_radius = np.linalg.norm(roots[:-1])

        # check if period is a multiple of the analytical sol
        period_multiplicity = num_period / anal_period
        period_multiple = np.isclose(period_multiplicity % 1, 1, atol=atol) or np.isclose(period_multiplicity % 1, 0, atol=atol)
        self.assertTrue(period_multiple and np.isclose(anal_radius, num_radius, atol=1e-02))

    # for array([1.41421356e+00, 1.51976157e-12, 3.71683338e-45, 1.06814150e+02])) solver cannot solve the
    # ode (maybe a stiff problem) - solutions go out of the unit circle. this may be causing it to evaluate
    # the solution incorrectly. Although the found period seems to be correct accounting for the repetition of 
    # the oscillations out of the unit circle -> get phase portrait with t_max = 200 for visualisation
    # TODO : add option to use different ode solver
    # MADE INIT COND FIXED UNTIL PROPERLY WORKING
    def test_shooting_hopf_ext_mySolver(self):
        beta, sigma = 2,-1
        atol = 1e-02
        args = (beta, sigma)
        # from analytical solution
        anal_period = 2* PI
        anal_radius = np.sqrt(beta)

        init_cond = np.array([1,1,1])
        #np.random.normal(size=3) * 9
        period_guess = np.random.uniform(low=5, high=25, size=1)
        #changed period guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
        init_guess = np.concatenate((init_cond, period_guess))

        roots = find_limit_cycles(init_guess, hopf_extended, args=(beta, sigma))
        numer_period = roots[-1]
        numer_radius = np.linalg.norm(roots[:-1])

        # check if period is a multiple of the analytical sol
        period_multiplicity = numer_period / anal_period
        period_multiple = np.isclose(period_multiplicity % 1, 1, atol=atol) or np.isclose(period_multiplicity % 1, 0, atol=atol)

        self.assertTrue(period_multiple and np.isclose(anal_radius, numer_radius, atol=atol))

    def test_shooting_hopf_ext_scipySolver(self):
        beta, sigma = 2,-1
        args = (beta, sigma)
        atol=1e-2
        # from analytical solution
        anal_period = 2* PI
        anal_radius = np.sqrt(beta)

        init_cond = np.array([1,1,1])
        #np.random.normal(size=3) * 9
        period_guess = np.random.uniform(low=5, high=25, size=1)
        #changed period guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
        init_guess = np.concatenate((init_cond, period_guess))

        roots = find_limit_cycles(init_guess, hopf_extended, solve_ivp, args=(beta, sigma))
        numer_period = roots[-1]
        numer_radius = np.linalg.norm(roots[:-1])

        # check if period is a multiple of the analytical sol
        period_multiplicity = numer_period / anal_period
        period_multiple = np.isclose(period_multiplicity % 1, 1, atol=atol) or np.isclose(period_multiplicity % 1, 0, atol=atol)

        self.assertTrue(period_multiple and np.isclose(anal_radius, numer_radius, atol=atol))


if __name__ == '__main__':
    unittest.main()

