from cmath import pi
import unittest
import numpy as np
from numerical_continuation import find_limit_cycles

PI = np.pi

def hopf_bifurcation(t, U, beta, sigma):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 + sigma *u1 *(u1*u1 + u2*u2)
    u2_dot = u1 + beta*u2 + sigma *u2 *(u1*u1 + u2*u2)
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

class TestShootingMethods(unittest.TestCase):

    #TODO has problems with array([-19.97797856, -16.40312613,   6.50074   ]) just returns this init guess (tried with T = 10)
    # probably to do with fsolve failing 
    # with T=10 finds sol = array([-1.41421356e+00, -7.02291778e-11,  1.13097336e+02])
    # where this is a valid solution
    def test_hopf(self, hopf_bifurcation):
        beta, sigma = 2,-1
        args = (beta, sigma)
        # from analytical solution
        anal_period = 2* np.pi
        anal_radius = np.sqrt(beta)

        init_cond = np.random.normal(size=2) * 9
        period_guess = np.random.uniform(low=5, high=25, size=1)
        #changed oeriod guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
        init_guess = np.concatenate((init_cond, period_guess))

        roots = find_limit_cycles(init_guess, hopf_bifurcation, args=(beta, sigma))
        num_period = roots[-1]
        num_radius = np.linalg.norm(roots[:-1])

        # check if period is a multiple of the analytical sol
        period_multiplicity = num_period / anal_period
        self.assertTrue(np.isclose(anal_period, num_period) and np.isclose(anal_radius, num_radius))


if __name__ == '__main__':
    unittest.main()

