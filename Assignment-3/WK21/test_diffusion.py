from doctest import testfile
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

import numpy as np
from solve_heat_eq2 import *
from scipy.integrate import solve_ivp
from math import pi

import unittest


class TestStringMethods(unittest.TestCase):

# TEST FOR HIGHER DIMENSIONAL SYS OF ODES
    def test_forw_euler_diffusion(self):
        rtol = 1e-02

        kappa = 1.0   # diffusion constant
        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for

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
        kwargs_fe = {'method' : 'Matrix'}

        u_fe = forw_eul_heat_eq(*args,**kwargs_fe)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))


    def test_back_euler_diffusion(self):
        rtol = 1e-02

        kappa = 1.0   # diffusion constant
        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for

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
        kwargs = dict()     # {'method' : 'matrix inversion'}

        u = back_eul_heat_eq(*args,**kwargs)
        rel_error = mean_rel_error(u[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))


    def test_crank_nicol_diffusion(self):
        rtol = 1e-02

        kappa = 1.0   # diffusion constant
        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for

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
        kwargs = {'method' : 'Matrix'}

        u = forw_eul_heat_eq(*args,**kwargs)
        rel_error = mean_rel_error(u[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))


if __name__ == '__main__':
    unittest.main()