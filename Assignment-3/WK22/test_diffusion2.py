from doctest import testfile
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

import numpy as np
from solve_heat_eq3 import *
from solve_diffusionV1 import *
from scipy.integrate import solve_ivp
from math import pi

import unittest


class TestStringMethods(unittest.TestCase):

# TEST FOR HIGHER DIMENSIONAL SYS OF ODES
    def test_forw_euler_diffusion(self):

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for

        def kappa(x):
            kappa = 1.0
            return kappa

        def u_I(x):
            # initial temperature distribution
            y = np.sin(pi*x/L)
            return y
        def u_exact(x,t):
            # the exact solution
            y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
            return y
        
        mx = 100
        mt = 10000

        x = np.linspace(0, L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time

        anal_u = u_exact(x,T)
        args = (t,x,u_I,kappa)

        u_fe = forw_eul_diffusion(*args)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))



    def test_forw_eul_matrix_kappaVariable_tx(self):
        ''' Generates A_FE matrix for a variable diffusion coefficient, dependent on x AND t. Compares result to a 
        known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 2x - t 
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(t,x):
            k =2*x - t
            return k

        t,x = get_grid_space(T,L,mt,mx) 
        deltax = x[1] - x[0]            # gridspacing in x
        deltat = t[1] - t[0]

        # define true matrix
        A_true = np.zeros((mt, mx-1,mx-1))
        A_true[0,0,:2] = np.array([1, 5])
        A_true[0,1,:3] = np.array([-5/3, -17/3,25/3])
        A_true[0,2,1:4] = np.array([5/3,-37/3,35/3])
        A_true[0,3,2:] = np.array([5,-19])

        A_true[1,0,:2] = np.array([59/9,20/9])
        A_true[1,1,:3] = np.array([-40/9,-1/9,50/9])
        A_true[1,2,1:4] = np.array([-10/9,-61/9,80/9])
        A_true[1,3,2:] = np.array([20/9,-121/9])

        A_true[2,0,:2] = np.array([109/9,-5/9])
        A_true[2,1,:3] = np.array([-65/9,49/9,25/9])
        A_true[2,2,1:4] = np.array([-35/9,-11/9,55/9])
        A_true[2,3,2:] = np.array([-5/9,-71/9])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_tx(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))

    def test_forw_eul_matrix_kappaConstant_tx(self):
        ''' Generates A_FE matrix for a constant diffusion coefficient, where the diffusion constant is defined as a function, dependent
        on x and t, returning a constant value. Compares result to a known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 1 
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(t, x):
            k = 1.0
            return k

        t,x = get_grid_space(T,L,mt,mx) 
        deltax = x[1] - x[0]            # gridspacing in x
        deltat = t[1] - t[0]
        lmbda = 1*deltat/(deltax**2)  # kappa is 1

        # define true matrix
        #tridiag(lambda, 1-2lambda, lambda)
        A_true = np.zeros((mt, mx-1, mx-1))
        A_true[:,0,:2] = np.array([1-2*lmbda,lmbda])
        A_true[:,1,:3] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,2,1:4] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,3,2:] = np.array([lmbda, 1-2*lmbda])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_tx(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))

    def test_forw_eul_matrix_kappaVariable_x(self):
        ''' Generates A_FE matrix for a variable diffusion coefficient, dependent on x. Compares result to a 
        known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 2x
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(x):
            k =2*x
            return k

        t,x = get_grid_space(T,L,mt,mx) 
        deltax = x[1] - x[0]            # gridspacing in x
        deltat = t[1] - t[0]

        # define true matrix
        A_true = np.zeros((mx-1,mx-1))
        A_true[0,:2] = np.array([1, 5])
        A_true[1,:3] = np.array([-5/3, -17/3,25/3])
        A_true[2,1:4] = np.array([5/3,-37/3,35/3])
        A_true[3,2:] = np.array([5,-19])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_x(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))

    def test_forw_eul_matrix_kappaConstant_x(self):
        ''' Generates A_FE matrix for a constant diffusion coefficient, where kappa is a function dependent on x but returning a constant. 
        Compares result to a known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 1 
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(x):
            k = 1.0
            return k

        t,x = get_grid_space(T,L,mt,mx) 
        deltax = x[1] - x[0]            # gridspacing in x
        deltat = t[1] - t[0]
        lmbda = 1*deltat/(deltax**2)  # kappa is 1

        # define true matrix
        #tridiag(lambda, 1-2lambda, lambda)
        A_true = np.zeros((mx-1, mx-1))
        A_true[0,:2] = np.array([1-2*lmbda,lmbda])
        A_true[1,:3] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[2,1:4] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[3,2:] = np.array([lmbda, 1-2*lmbda])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_x(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))


if __name__ == '__main__':
    unittest.main()