from doctest import testfile
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

import numpy as np
from solve_diffusionV3 import *

import unittest

# SYNTAX python -m unittest -v test_module

class TestStringMethods(unittest.TestCase):

    def test_simple_heat_eq_forw_euler(self):
        ''' Tests the numerical solution of 
        u_t = D*u_xx    for u(t=0, x) = sin(pi*x/L) 
        and 0 dirichlet boundary conditions.
        
        Against the analytical solution:
        exp(t*-D*pi^2/L^2)*sin(pi*x/L)

        where D (kappa) is given as a constant, and L : length of spacial domain
        TESTING WHETHER FORWARD EULER CAN SOLVE SIMPLE HEAT EQUATION
        '''
        rtol=1e-02

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for
        def kappa(x):
            return 1.0

        def u_I(x):
            # initial temperature distribution
            y = np.sin(pi*x/L)
            return y
        def u_exact(x,t):
            # the exact solution
            y = np.exp(-1.0*(pi**2/L**2)*t)*np.sin(pi*x/L)
            return y
        
        mx = 100
        mt = 10000

        t, x = get_grid_space(T,L,mt,mx)

        anal_u = u_exact(x,T)

        args = (t,x,kappa,u_I)
        kwargs = {'discretization' : 'Forward Euler'}

        u_fe = solve_diffusion(*args,**kwargs)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))