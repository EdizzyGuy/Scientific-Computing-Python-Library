from doctest import testfile
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

import numpy as np
from solve_heat_eq3 import *
from solve_diffusionV3 import *
from scipy.integrate import solve_ivp
from math import pi

import unittest

# SYNTAX python -m unittest -v test_module

class TestStringMethods(unittest.TestCase):

    def test_suitable_mt(self):
        ''' Test that the suitable mt is optimal '''
        T, L = [0,0.5], [0,1.0]
        kappa = 1.0
        mx = 100
        
        mt = find_suitable_mt(mx,kappa,L,T)
        self.assertTrue(mt == 10000)
    
    def test_suitable_mx(self):
        ''' Test that the suitable mx calculated is optimal '''
        T, L = [0, 0.5], [0, 1.0]
        kappa = 1.0
        mt = 10000
        
        mx = find_suitable_mx(mt,kappa,L,T)
        self.assertTrue(mx == 100)

    def test_grid_space(self):
        ''' Test the correct gridspace is outputted from get_grid_space '''
        T, L = [0, 2], [0, 4]
        mt, mx = 2, 4

        true_x = np.array([0,1,2,3,4])
        true_t = np.array([0,1,2])

        t,x = get_grid_space(T,L,mt,mx)
        self.assertTrue(np.all(np.isclose(true_x, x)) and np.all(np.isclose(true_t, t)))
    
    def test_grid_spacing(self):
        ''' Test that function can accurately find deltat and deltax '''
        t, x = np.array([0,1,2]), np.array([0,1,2,3,4])

        deltat, deltax = get_grid_spacing(t,x)
        self.assertTrue(deltax == 1 and deltat == 1)
    
    def test_forw_eul_matrix_DoubleDirichlet(self):
        ''' Will calculate the Forward Euler matrix used in the pde solver, for Dirichlet boundaries at both ends.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Dirichlet',), ('Dirichlet',)]
        true_mat = np.array([[1-2,3/2,0],
                            [3/2,1-4,5/2],
                            [0,5/2,1-6]])

        forw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Forward Euler')
        forw_euler_matrix = forw_euler_sparse.toarray()
        validity = np.all(np.isclose(forw_euler_matrix, true_mat))
        
        self.assertTrue(validity)

    def test_forw_eul_matrix_DirichletNeumann(self):
        ''' Will calculate the Forward Euler matrix used in the pde solver, for Dirichlet boundary condition at the left and Neumann at the right.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Dirichlet',), ('Neumann',)]
        true_mat = np.array([[1-2,3/2,0,0],
                            [3/2,1-4,5/2,0],
                            [0,5/2,1-6,7/2],
                            [0,0,8,1-8]])

        forw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Forward Euler')
        forw_euler_matrix = forw_euler_sparse.toarray()
        validity = np.all(np.isclose(forw_euler_matrix, true_mat))
            
        self.assertTrue(validity)

    def test_forw_eul_matrix_NeumannDirichlet(self):
        ''' Will calculate the Forward Euler matrix used in the pde solver, for Neumann boundary condition at the left and Dirichlet at the right.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Neumann',), ('Dirichlet',)]
        true_mat = np.array([[1,0,0,0],
                            [1/2,1-2,3/2,0],
                            [0,3/2,1-4,5/2],
                            [0,0,5/2,1-6]])

        forw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Forward Euler')
        forw_euler_matrix = forw_euler_sparse.toarray()
        validity = np.all(np.isclose(forw_euler_matrix, true_mat))
            
        self.assertTrue(validity)

    def test_forw_eul_matrix_DoubleNeumann(self):
        ''' Will calculate the Forward Euler matrix used in the pde solver, for Neumann boundaries at both ends.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Neumann',), ('Neumann',)]
        true_mat = np.array([[1,0,0,0,0],
                            [1/2,1-2,3/2,0,0],
                            [0,3/2,1-4,5/2,0],
                            [0,0,5/2,1-6,7/2],
                            [0,0,0,8,1-8]])

        forw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Forward Euler')
        forw_euler_matrix = forw_euler_sparse.toarray()
        validity = np.all(np.isclose(forw_euler_matrix, true_mat))
        
        self.assertTrue(validity)

    def test_forw_eul_matrix_Periodic(self):
        ''' Will calculate the Forward Euler matrix used in the pde solver, for periodic boundary conditions.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Periodic',)]
        true_mat = np.array([[1,1/2,0,-0.5],
                            [1/2,1-2,3/2,0],
                            [0,3/2,1-4,5/2],
                            [7/2,0,5/2,1-6]])

        forw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Forward Euler')
        forw_euler_matrix = forw_euler_sparse.toarray()
        validity = np.all(np.isclose(forw_euler_matrix, true_mat))
            
        self.assertTrue(validity)

    def test_backw_eul_matrix_DoubleDirichlet(self):
        ''' Will calculate the Backward Euler matrix used in the pde solver, for Dirichlet boundaries at both ends.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Dirichlet',), ('Dirichlet',)]
        true_mat = np.array([[1+4,-5/2,0],
                            [-5/2,1+6,-7/2],
                            [0,-7/2,1+8]])

        backw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Backward Euler')
        backw_euler_matrix = backw_euler_sparse.toarray()
        validity = np.all(np.isclose(backw_euler_matrix, true_mat))

        self.assertTrue(validity)

    def test_backw_eul_matrix_DirichletNeumann(self):
        ''' Will calculate the Backward Euler matrix used in the pde solver, for Dirichlet boundary condition at left end and Neumann at right end.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Dirichlet',), ('Neumann',)]
        true_mat = np.array([[1+4,-5/2,0,0],
                            [-5/2,1+6,-7/2,0],
                            [0,-7/2,1+8,-9/2],
                            [0,0,-10,1+10]])

        backw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Backward Euler')
        backw_euler_matrix = backw_euler_sparse.toarray()
        validity = np.all(np.isclose(backw_euler_matrix, true_mat))
        self.assertTrue(validity)

    def test_backw_eul_matrix_NeumannDirichlet(self):
        ''' Will calculate the Backward Euler matrix used in the pde solver, for Neumann boundary condition at left end and Dirichlet at right end.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Neumann',), ('Dirichlet',)]
        true_mat = np.array([[1+2,-2,0,0],
                            [-3/2,1+4,-5/2,0],
                            [0,-5/2,1+6,-7/2],
                            [0,0,-7/2,1+8]])

        backw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Backward Euler')
        backw_euler_matrix = backw_euler_sparse.toarray()
        validity = np.all(np.isclose(backw_euler_matrix, true_mat))
        self.assertTrue(validity)

    def test_backw_eul_matrix_DoubleNeumann(self):
        ''' Will calculate the Backward Euler matrix used in the pde solver, for Neumann boundary conditions at both ends.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Neumann',), ('Neumann',)]
        true_mat = np.array([[1+2,-2,0,0,0],
                            [-3/2,1+4,-5/2,0,0],
                            [0,-5/2,1+6,-7/2,0],
                            [0,0,-7/2,1+8,-9/2],
                            [0,0,0,-10,1+10]])

        backw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Backward Euler')
        backw_euler_matrix = backw_euler_sparse.toarray()
        validity = np.all(np.isclose(backw_euler_matrix, true_mat))          
        self.assertTrue(validity)

    def test_backw_eul_matrix_Periodic(self):
        ''' Will calculate the Backward Euler matrix used in the pde solver, for periodic boundary conditions.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Periodic',)]
        true_mat = np.array([[1+2,-3/2,0,-0.5],
                            [-3/2,1+4,-5/2,0],
                            [0,-5/2,1+6,-7/2],
                            [-9/2,0,-7/2,1+8]])

        backw_euler_sparse = get_pde_solving_matrix(t,x,kappa,boundary_conditions,discretization='Backward Euler')
        backw_euler_matrix = backw_euler_sparse.toarray()
        validity = np.all(np.isclose(backw_euler_matrix, true_mat))        
        self.assertTrue(validity)

    def test_crank_nicholson_matrices_DoubleDirichlet(self):
        ''' Will calculate the Crank Nicholson matrix used in the pde solver, for Dirichlet boundary conditions at each end. 
        Checks result against a known result'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Dirichlet',), ('Dirichlet',)]
        true_mat_A = np.array([[1+3/2,-1,0],
                                [-1,1+5/2,-3/2],
                                [0,-3/2,1+7/2]])
        true_mat_B =  np.array([[1-3/2,1,0],
                                [1,1-5/2,3/2],
                                [0,3/2,1-7/2]])

        CN_sparse_A, CN_sparse_B = get_pde_solving_matrix(t,x,kappa,boundary_conditions, discretization='Crank Nicholson')
        CN_A, CN_B= CN_sparse_A.toarray(), CN_sparse_B.toarray()
        validity = np.all(np.isclose(CN_A, true_mat_A)) and np.all(np.isclose(CN_B, true_mat_B))       
        self.assertTrue(validity)

    def test_crank_nicholson_matrices_DirichletNeumann(self):
        ''' Will calculate the Crank Nicholson matrix used in the pde solver, for Dirichlet boundary condition at left end and Neumann at right end.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Dirichlet',), ('Neumann',)]
        true_mat_A = np.array([[1+3/2,-1,0,0],
                            [-1,1+5/2,-3/2,0],
                            [0,-3/2,1+7/2,-2],
                            [0,0,-9/2,1+9/2]])
        true_mat_B =  np.array([[1-3/2,1,0,0],
                            [1,1-5/2,3/2,0],
                            [0,3/2,1-7/2,2],
                            [0,0,9/2,1-9/2]])

        CN_sparse_A, CN_sparse_B = get_pde_solving_matrix(t,x,kappa,boundary_conditions, discretization='Crank Nicholson')
        CN_A, CN_B= CN_sparse_A.toarray(), CN_sparse_B.toarray()
        validity = np.all(np.isclose(CN_A, true_mat_A)) and np.all(np.isclose(CN_B, true_mat_B))       
        self.assertTrue(validity)

    def test_crank_nicholson_matrices_NeumannDirichlet(self):
        ''' Will calculate the Crank Nicholson matrix used in the pde solver, for Neumann boundary condition at left end and Dirichlet at right end.
        Check result against a known result.'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Neumann',), ('Dirichlet',)]
        true_mat_A = np.array([[1+1/2,-1/2,0,0],
                            [-1/2,1+3/2,-1,0],
                            [0,-1,1+5/2,-3/2],
                            [0,0,-3/2,1+7/2]])
        true_mat_B =  np.array([[1-1/2,1/2,0,0],
                            [1/2,1-3/2,1,0],
                            [0,1,1-5/2,3/2],
                            [0,0,3/2,1-7/2]])
                            
        CN_sparse_A, CN_sparse_B = get_pde_solving_matrix(t,x,kappa,boundary_conditions, discretization='Crank Nicholson')
        CN_A, CN_B= CN_sparse_A.toarray(), CN_sparse_B.toarray()
        validity = np.all(np.isclose(CN_A, true_mat_A)) and np.all(np.isclose(CN_B, true_mat_B))         
        self.assertTrue(validity)

    def test_crank_nicholson_matrices_DoubleNeumann(self):
        ''' Will calculate the Crank Nicholson matrix used in the pde solver, for Neumann boundary conditions at each end.
        Checks result against a known result'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Neumann',), ('Neumann',)]
        true_mat_A = np.array([[1+1/2,-1/2,0,0,0],
                                [-1/2,1+3/2,-1,0,0],
                                [0,-1,1+5/2,-3/2,0],
                                [0,0,-3/2,1+7/2,-2],
                                [0,0,0,-9/2,1+9/2]])
        true_mat_B =  np.array([[1-1/2,1/2,0,0,0],
                                [1/2,1-3/2,1,0,0],
                                [0,1,1-5/2,3/2,0],
                                [0,0,3/2,1-7/2,2],
                                [0,0,0,9/2,1-9/2]])
                            
        CN_sparse_A, CN_sparse_B = get_pde_solving_matrix(t,x,kappa,boundary_conditions, discretization='Crank Nicholson')
        CN_A, CN_B= CN_sparse_A.toarray(), CN_sparse_B.toarray()
        validity = np.all(np.isclose(CN_A, true_mat_A)) and np.all(np.isclose(CN_B, true_mat_B))         
        self.assertTrue(validity)

    def test_crank_nicholson_matrices_Periodic(self):
        ''' Will calculate the Crank Nicholson matrix used in the pde solver, for periodic boundary conditions, and check result against a known result'''
        mx, L = 4, 4
        mt, T = 2, 2

        def kappa(t,x):
            return x + t

        t,x = get_grid_space(T,L,mt,mx)

        boundary_conditions = [('Periodic',)]
        true_mat_A = np.array([[1+1/2,-1/2,0,0],
                                [-1/2,1+3/2,-1,0],
                                [0,-1,1+5/2,-3/2],
                                [-2,0,-3/2,1+7/2]])
        true_mat_B =  np.array([[1-1/2,1/2,0,0],
                                [1/2,1-3/2,1,0],
                                [0,1,1-5/2,3/2],
                                [2,0,3/2,1-7/2]])
                            
        CN_sparse_A, CN_sparse_B = get_pde_solving_matrix(t,x,kappa,boundary_conditions, discretization='Crank Nicholson')
        CN_A, CN_B= CN_sparse_A.toarray(), CN_sparse_B.toarray()
        validity = np.all(np.isclose(CN_A, true_mat_A)) and np.all(np.isclose(CN_B, true_mat_B))         
        self.assertTrue(validity)

