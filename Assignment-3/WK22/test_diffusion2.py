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
    def test_forw_euler_diffusion_nohomo_dirich(self):
        '''Tests the numerical solution of 
        u_t = u_xx + x    for u(t=0, x) = 3*sin(2*pi*x/L) + 2(1-x)
        and inhomogenous dirichlet boundary conditions : u(0,t) = 2; u(L,t) = t
        
        Against the analytical solution:
        u(t,x) = 3*sin(2*pi*x)*exp(-4*pi^2*t) +2 + (t-2)*x

        Where L = length of spacial domain, 
        TESTING WHETHER FORWARD EULER CAN HANDLE NON HOMOGENOUS DIRICHLET BOUNDARIES
        '''
        rtol=1e-02

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for
        kappa = 1.0    

        def u_I(x):
            # initial temperature distribution
            y = 3*np.sin(2*pi*x/L)
            return y
        def rhs_func(t,x):
            f = x
            return f
        def u_exact(x,t):
            # the exact solution
            y = 3*np.sin(2*pi*x/L)*np.exp(-4*kappa*pi**2*t/L**2) +2 + (t-2)*x
            return y
        #u(0,t) = 2; u(L,t) = t
        l_bound = lambda t : 2
        r_bound = lambda t : t
        
        mx = 100
        mt = 10000

        t, x = get_grid_space(T,L,mt,mx)     # mesh points in space and time

        anal_u = u_exact(x,T)
        args = (t,x,u_I,kappa)
        kwargs = {'l_boundary':l_bound,'r_boundary':r_bound,'rhs_func': rhs_func}

        u_fe = forw_eul_diffusion_use_matrix(*args, **kwargs)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))

    def test_forw_euler_diffusion_rhsFUNC_tx(self):
        '''Tests the numerical solution of 
        u_t = D*u_xx + f(t,x)    for u(t=0, x) = sin(2*pi*x/L) and f(t,x) = exp(t*-D*pi^2/L^2)*sin(pi*x/L)
        and 0 dirichlet boundary conditions.
        
        Against the analytical solution:
        u(t,x) = L^2/(D*pi^2)*(1-exp(-D*pi^2*t/L^2))*sin(pi*x/L) + exp(-4*D*(pi^2/L^2)*t)*sin(2*pi*x/L)

        Where D : diffusion constant, L = length of spacial domain, 
        TESTING WHETHER FORWARD EULER CAN HANDLE A RHS FUNCTION DEPENDANT ON X
        '''
        rtol=1e-02

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for
        kappa = 1.0

        def u_I(x):
            # initial temperature distribution
            y = np.sin(2*pi*x/L)
            return y
        def rhs_func(t,x):
            f = np.exp(-t*kappa*pi**2/L**2)*np.sin(pi*x/L)
            return f
        def u_exact(x,t):
            # the exact solution
            y = t*np.sin(pi*x/L)*np.exp(-kappa*pi**2*t/L**2) + np.exp(-4*kappa*pi**2*t/L**2)*np.sin(2*pi*x/L)
            return y
        
        mx = 100
        mt = 10000

        x = np.linspace(0, L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time

        anal_u = u_exact(x,T)
        args = (t,x,u_I,kappa)
        kwargs = {'rhs_func': rhs_func}

        u_fe = forw_eul_diffusion_use_matrix(*args, **kwargs)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))

    def test_forw_euler_diffusion_rhsFUNC_x(self):
        '''Tests the numerical solution of 
        u_t = D*u_xx + f(x)    for u(t=0, x) = sin(2*pi*x/L) and f(x) = sin(pi*x/L)
        and 0 dirichlet boundary conditions.
        
        Against the analytical solution:
        u(t,x) = L^2/(D*pi^2)*(1-exp(-D*pi^2*t/L^2))*sin(pi*x/L) + exp(-4*D*(pi^2/L^2)*t)*sin(2*pi*x/L)
        
        Where D : diffusion constant, L = length of spacial domain,
        TESTING WHETHER FORWARD EULER CAN HANDLE A RHS FUNCTION DEPENDANT ON X
        '''
        rtol=1e-02

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for
        kappa = 1.0

        def u_I(x):
            # initial temperature distribution
            y = np.sin(2*pi*x/L)
            return y
        def rhs_func(t,x):
            f = np.sin(pi*x/L)
            return f
        def u_exact(x,t):
            # the exact solution
            y = L**2/(kappa*pi**2)*(1-np.exp(-kappa*pi**2*t/L**2))*np.sin(pi*x/L) + np.exp(-4*kappa*(pi**2/L**2)*t)*np.sin(2*pi*x/L)
            return y
        
        mx = 100
        mt = 10000

        x = np.linspace(0, L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time

        anal_u = u_exact(x,T)
        args = (t,x,u_I,kappa)
        kwargs = {'rhs_func': rhs_func}

        u_fe = forw_eul_diffusion_use_matrix(*args,**kwargs)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))

    def test_forw_euler_diffusion_kappaCONST_FUNC_tx(self):
        ''' Tests the numerical solution of 
        u_t = D*u_xx    for u(t=0, x) = sin(pi*x/L)
        
        Against the analytical solution:
        exp(t*-D*pi^2/L^2)*sin(pi*x/L)

        Where D (kappa) is given as a function of x and t, and returns a constant value, and L : length of spacial domain 
        TESTING WHETHER FORWARD EULER CAN HANDLE KAPPA FUNCTIONS OF X AND T
        '''
        rtol=1e-02

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for
        
        def kappa(t,x):
            kappa = 1.0
            return kappa
        def u_I(x):
            # initial temperature distribution
            y = np.sin(pi*x/L)
            return y
        def u_exact(x,t):
            # the exact solution
            k = 1.0
            y = np.exp(-k*(pi**2/L**2)*t)*np.sin(pi*x/L)
            return y
        
        mx = 100
        mt = 10000

        x = np.linspace(0, L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time

        anal_u = u_exact(x,T)
        args = (t,x,u_I,kappa)

        u_fe = forw_eul_diffusion_use_matrix(*args)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))

    def test_forw_euler_diffusion_kappaCONST_FUNC_x(self):
        ''' Tests the numerical solution of 
        u_t = D*u_xx    for u(t=0, x) = sin(pi*x/L)
        
        Against the analytical solution:
        exp(t*-D*pi^2/L^2)*sin(pi*x/L)

        where D (kappa) is given as a function of x and returns a constant value, and L : length of spacial domain
        TESTING WHETHER FORWARD EULER CAN HANDLE KAPPA FUNCTIONS OF X
        '''
        rtol=1e-02

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
            k = 1.0
            y = np.exp(-k*(pi**2/L**2)*t)*np.sin(pi*x/L)
            return y
        
        mx = 100
        mt = 10000

        x = np.linspace(0, L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time

        anal_u = u_exact(x,T)
        args = (t,x,u_I,kappa)

        u_fe = forw_eul_diffusion_use_matrix(*args)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))

    def test_forw_euler_diffusion_kappaCONST(self):
        ''' Tests the numerical solution of 
        u_t = D*u_xx    for u(t=0, x) = sin(pi*x/L) and 0 dirichlet boundary conditions.
        
        Against the analytical solution:
        exp(t*-D*pi^2/L^2)*sin(pi*x/L)

        where D (kappa) is given as a constant, and L : length of spacial domain
        TESTING WHETHER FORWARD EULER CAN SOLVE SIMPLE HEAT EQUATION
        '''
        rtol=1e-02

        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for
        kappa = 1.0

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

        u_fe = forw_eul_diffusion_use_matrix(*args)
        rel_error = mean_rel_error(u_fe[-1], anal_u)

        self.assertTrue(np.isclose(rel_error, 1, rtol=rtol))

    def test_forw_eul_matrix_kappaVariable_tx(self):
        ''' Generates A_FE matrix for a VARIABLE diffusion coefficient, DEPENDANT ON X AND T. Compares result to a 
        known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 2x - t 
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(t,x):
            k =2*x - t
            return k

        t,x = get_grid_space(T,L,mt,mx) 

        # define true matrix
        A_true = np.zeros((mt, mx+1,mx+1))
        A_true[0,1,:3] = np.array([-5/3, -17/3,25/3])
        A_true[0,1,1:4] = np.array([5/3,-37/3,35/3])
        A_true[0,2,2:5] = np.array([5,-19,15])
        A_true[0,3,3:] = np.array([5,-19,55/3])

        A_true[1,1,:3] = np.array([-40/9,-1/9,50/9])
        A_true[1,2,1:4] = np.array([-10/9,-61/9,80/9])
        A_true[1,3,2:5] = np.array([20/9,-121/9, 110/9])
        A_true[1,4,3:] = np.array([50/9,-181/9, 140/9])

        A_true[2,1,:3] = np.array([-65/9,49/9,25/9])
        A_true[2,2,1:4] = np.array([-35/9,-11/9,55/9])
        A_true[2,3,2:5] = np.array([-5/9,-71/9, 85/9])
        A_true[2,4,3:] = np.array([-5/9,-71/9, 115/9])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_tx(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))

    def test_forw_eul_matrix_kappaConstant_tx(self):
        ''' Generates A_FE matrix for a CONSTANT diffusion coefficient, where the diffusion constant is defined as a function, DEPENDANT ON 
        X AND T, returning a constant value. Compares result to a known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 1 
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(t, x):
            k = 1.0
            return k

        t,x = get_grid_space(T,L,mt,mx) 
        deltat, deltax = get_grid_spacing(t,x)            # gridspacing in x and t
        lmbda = 1*deltat/(deltax**2)  # kappa is 1

        # define true matrix
        #tridiag(lambda, 1-2lambda, lambda)
        A_true = np.zeros((mt, mx+1, mx+1))
        A_true[:,1,:3] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,2,1:4] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,3,2:5] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,4,3:] = np.array([lmbda, 1-2*lmbda,lmbda])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_tx(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))

    def test_forw_eul_matrix_kappaVariable_x(self):
        ''' Generates A_FE matrix for a VARIABLE diffusion coefficient, DEPENDANT ON X. Compares result to a 
        known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 2x
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(x):
            k =2*x
            return k

        t,x = get_grid_space(T,L,mt,mx)  

        # define true matrix
        A_true = np.zeros((mt, mx+1,mx+1))
        A_true[:,1,:3] = np.array([-5/3, -17/3,25/3])
        A_true[:,1,1:4] = np.array([5/3,-37/3,35/3])
        A_true[:,2,2:5] = np.array([5,-19,15])
        A_true[:,3,3:] = np.array([5,-19,55/3])

        # get matrix from predefined function
        A = forw_eul_pde_matrix_varKappa_x(t,x,kappa)

        self.assertTrue(np.all(np.isclose(A_true,A)))

    def test_forw_eul_matrix_kappaConstant_x(self):
        ''' Generates A_FE matrix for a CONSTANT diffusion coefficient, where kappa is a function DEPENDANT ON X but returning a constant. 
        Compares result to a known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 1 
        '''
        T,L,mx,mt = 1,1,5,3

        def kappa(x):
            k = 1.0
            return k

        t,x = get_grid_space(T,L,mt,mx) 
        deltat, deltax = get_grid_spacing(t,x)  
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

    def test_forw_eul_matrix_kappaConstant_x(self):
        ''' Generates A_FE matrix for a CONSTANT diffusion coefficient, where kappa is a constant. 
        Compares result to a known (hand calculated) A_FE and asserts that they are the same.
        '''
        T,L,mx,mt = 1,1,5,3
        kappa = 1.0

        t,x = get_grid_space(T,L,mt,mx) 
        deltat, deltax = get_grid_spacing(t,x)  
        lmbda = 1*deltat/(deltax**2)  # kappa is 1

        # define true matrix
        #tridiag(lambda, 1-2lambda, lambda)
        A_true = np.zeros((1,mx-1, mx-1))
        A_true[:,0,:2] = np.array([1-2*lmbda,lmbda])
        A_true[:,1,:3] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,2,1:4] = np.array([lmbda, 1-2*lmbda,lmbda])
        A_true[:,3,2:] = np.array([lmbda, 1-2*lmbda])

        # get matrix from predefined function
        A = forw_eul_pde_matrix(lmbda, mx)

        self.assertTrue(np.all(np.isclose(A_true,A)))


if __name__ == '__main__':
    unittest.main()