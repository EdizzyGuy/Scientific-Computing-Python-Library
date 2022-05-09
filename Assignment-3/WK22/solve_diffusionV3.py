import numpy as np
from scipy import linalg
from scipy import sparse
import inspect
import time


# MAKE A SPECIAL CASE IN EULER FOR WHEN VAR KAPPA IS DEP ONLY ON T

'''
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

PUT U_I ARGS ON ALL FUNCS
LET BACKWARD EULER PICK FROM SCIPY LINALG SOLVE AND MAT INV

'''

def find_suitable_mt(mx, kappa, L, T, max_lambda=0.5):
    ''' Finds a suitable number of gridpoints in time, given a number of gridpoints in space, such that lambda is less than, or equal to, max_lambda.
    max_lambda defaults to the value such that the forward Euler stability criterion is met. 
    Kappa must be a constant value for this value of mt to be valid.
    Args:
        mx (int)           : Number of gridpoints in space.
        kappa (float)      : Value of the diffusion CONSTANT
        L (list)           : start and end points of the domain in x
        T (list)           : start and end points of time to integrate for.
        max_lambda (float) : maximum value of lambda (:deltat/deltax^2) allowed.
    Returns:
        optimal_mx (int)   : maximum integer for mx such that lambda is less than or equal to max_lambda.
    '''
    x_range = np.max(L) - np.min(L)
    t_range = np.max(T) - np.min(T)

    mt = kappa * t_range * mx**2 /(max_lambda * x_range**2)
    optimal_mt = int(np.floor(mt))
    return optimal_mt

def find_suitable_mx(mt, kappa, L, T, max_lambda=0.5):
    ''' Finds a suitable number of gridpoints in space, given a number of gridpoints in time, such that lambda is less than, or equal to, max_lambda.
    max_lambda defaults to the value such that the forward Euler stability criterion is met. 
    Kappa must be a constant value for this value of mt to be valid.
    Args:
        mt (int)           : Number of gridpoints in time.
        kappa (float)      : Value of the diffusion CONSTANT
        L (list)           : start and end points of the domain in x
        T (list)           : start and end points of time to integrate for.
        max_lambda (float) : maximum value of lambda (:deltat/deltax^2) allowed.
    Returns:
        optimal_mt (int)   : maximum integer for mt such that lambda is less than or equal to max_lambda
    '''
    x_range = np.max(L) - np.min(L)
    t_range = np.max(T) - np.min(T)

    mx = np.sqrt(max_lambda * x_range**2 * mt /(kappa * t_range))
    optimal_mx = int(np.floor(mx))
    return optimal_mx

def get_grid_space(T, L, mt, mx):
    ''' Returns discretized gridspace of the domain in x and time
    Args:
        T (list/float)  : Two element list which signifies the start and the end of the domain in time. If given as a
                        float then start is assumed to be 0.
        L (list/float)  : Two element list which signifies the start and the end of the domain in 1D space. If given as
                        float then start is assumed to be 0.
        mt (int)        : number of grid points in time.
        mx (int)        : number of gridpoints in space
    Returns:
        t (np.ndarray) : discretized values of time in the grid space
        x (np.ndarray) : discretized values of x in the grid space
    '''
    # Set up the numerical environment variables
    if isinstance(T, float) or isinstance(L, int):
        T = [0, T]
    if isinstance(L, float) or isinstance(L, int):
        L = [0, L]
    x = np.linspace(L[0], L[1], mx+1)     # mesh points in space
    t = np.linspace(T[0], T[1], mt+1)     # mesh points in time           
    return t, x

def get_grid_spacing(t,x): 
    ''' Gets values of deltax and deltat from the gridspace
    Args:
        t (np.ndarray)      : discretized values in time of the grid space
        x (np.ndarray)      : discretuzed values of x in the grid space
    Returns:
        deltat (np.float64) : difference between successive elements in discretized t
        deltax (np.float64) : difference between successive elements in discretized x
    '''
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    return deltat, deltax

def get_pde_solving_matrix(t,x,kappa,boundary_conditions=[('Dirichlet',), ('Dirichlet',)], time_index=0, discretization='Crank Nicholson'):
    ''' Will generate the neccessary sparse matrix to solve the pde question numerically, dependent on boundary conditions, and the discretization
    that has been used. Valid boundary_conditions are ['Dirichlet', 'Neumann', 'Periodic'] and valid discretizations are 
    ['Forward Euler', 'Backward Euler', 'Crank Nicholson'].
    If no arguements provided will assume double Dirichlet boundary conditions using a crank nicholson discretization. If 'Periodic' is in boundary
    conditions the other elements will be ignored and the pde will be solved in a periodic domain.

    NOTE:
    The matrix obtained is used in the following operations:

    FORWARRD EULER:                     {A_FE @ u[j] = u[j+1]}
        Double Dirichlet        :       mat_vec_prod = pde_mat.dot(u[j,1:-1]); mat_vec_prod[[0,-1]] += lambda*[P(t[j]), Q(t[j])]; u[j+1,1:-1] = mat_vec_prod
        Dirichlet | Neumann     :       mat_vec_prod = pde_mat.dot(u[j,1:]); mat_vec_prod[[0,-1]] += lambda*[P(t[j]), 2*deltax*Q(t[j])]; u[j+1,1:] = mat_vec_prod
        Neumann   | Dirichlet   :       mat_vec_prod = pde_mat.dot(u[j,:-1]); mat_vec_prod[[0,-1]] += lambda*[-2*deltax*P(t[j]), Q(t[j])]; u[j+1,:-1] = mat_vec_prod
        Double Neumann          :       mat_vec_prod = pde_mat.dot(u[j]); mat_vec_prod[[0,-1]] += 2*deltax*lambda*[-P(t[j]), Q(t[j])]; u[j+1] = mat_vec_prod
        Periodic                :       mat_vec_prod = pde_mat.dot(u[j,:-1]); u[j+1,:-1] = mat_vec_prod

    BACKWARD EULER:                     {A_BE @ u[j+1] = u[j]}          WILL NEED TO SOLVE Ax = b (TDMA)
        Double Dirichlet        :       b = u[j,1:-1]; b[[0,-1]] += lambda*[P(t[j+1]), Q[t[j+1]]]; x = sparse.linalg.spsolve(pde_mat, b); u[j+1,1:-1] = x
        Dirichlet | Neumann     :       b = u[j,1:]; b[[0,-1]] += lambda*[P(t[j+1]), 2*deltax*Q[t[j+1]]]; x = sparse.linalg.spsolve(pde_mat, b); u[j+1,1:] = x
        Neumann   | Dirichlet   :       b = u[j,:-1]; b[[0,-1]] += lambda*[-2*deltax*P(t[j+1]), Q[t[j+1]]]; x = sparse.linalg.spsolve(pde_mat, b); u[j+1,:-1] = x
        Double Neumann          :       b = u[j]; b[[0,-1]] += 2*deltax*lambda*[-P(t[j+1]), Q[t[j+1]]]; x = sparse.linalg.spsolve(pde_mat, b); u[j+1,1:-1] = x
        Periodic                :       b = u[j,:-1]; x = sparse.linalg.spsolve(pde_mat, b); u[j+1,:-1] = x

    CRANK NICHOLSON                     {A_CN @ u[j+1] = B_CN @ u[j]}   WILL NEED TO SOLVE Ax = b (TDMA)  
        Double Dirichlet        :       b = pde_mat_B.dot(u[j,1:-1]); b[[0,-1]] += lambda/2*[P(t[j+1])+P(t[j]), Q(t[j+1])+Q(t[j])]; x = sparse.linalg.spsolve(pde_mat_A, b); u[j+1,1:-1] = x
        Dirichlet | Neumann     :       b = pde_mat_B.dot(u[j,1:]); b[[0,-1]] += lambda/2*[P(t[j+1])+P(t[j]), 2*deltax*(Q(t[j+1])+Q(t[j]))]; x = sparse.linalg.spsolve(pde_mat_A, b); u[j+1,1:] = x
        Neumann   | Dirichlet   :       b = pde_mat_B.dot(u[j,:-1]); b[[0,-1]] += lambda/2*[-2*deltax*(P(t[j+1])+P(t[j])), Q(t[j+1])+Q(t[j])]; x = sparse.linalg.spsolve(pde_mat_A, b); u[j+1,:-1] = x
        Double Neumann          :       b = pde_mat_B.dot(u[j]); b[[0,-1]] += deltax*lambda*[-(P(t[j+1])+P(t[j])), Q(t[j+1])+Q(t[j])]; x = sparse.linalg.spsolve(pde_mat_A, b); u[j+1] = x
        Periodic                :       b = pde_mat_B.dot(u[j,:-1]); x = sparse.linalg.spsolve(pde_mat_A, b); u[j+1,:-1] = x

=========================================================================================================================================================================================================    
    Args:
        t (np.ndarray)                          : discretized values in time of the grid space
        x (np.ndarray)                          : discretuzed values of x in the grid space
        boundary_conditions (list)                   : list of classifications of the boundary conditions on the 1D domain, such that the first index is the boundary condition
                                                on the left side of the domain, and the second index is the type of boundary condition on the right side of the domain.
        time_index (int)                        : value of j such that current time = t[j]. Used in the case that kappa is variable in t.
        discretization (str)                    : The kind of discretization used to solve the pde. This will affect the outputted matrix

    Returns:
        pde_matrix (scipy.sparse.dia_matrix)    : Sparse matrix for use in forward euler or backward euler.
        pde_matrix_A (scipy.sparse.dia_matrix)  : First matrix required for solving pde using crank nicholson
        pde_matrix_B (scipy.sparse.dia_matrix)  : Second matrix required for solving pde using crank nicholson
    '''
# dirichlet differences:
#   diag is (-lmbda, 1+2lmbda, lmbda)
#   kappa is taken at t[j+1]
    valid_boundary_conditions = ['Dirichlet', 'Neumann', 'Periodic']
    valid_discretizations = ['Crank Nicholson','Backward Euler','Forward Euler']

    assert all(condition[0] in valid_boundary_conditions for condition in boundary_conditions), \
        'One or more of the boundary conditions enetered is invalid. Exiting...'
    # if discretization arguement is not valid default to crank nicholson
    if discretization not in valid_discretizations:
        discretization = 'Crank Nicholson'
    
    deltat, deltax = get_grid_spacing(t,x)
    mt, mx = int(t[-1]/deltat), int(x[-1]/deltax)
    c = deltat/deltax**2
    
    # pow(-1,back_e) = 1 if back_e is 0 and -1 if back_e is 1
    # pow(2, -crank_n) = 1 if crank_n is 0 and 1/2 if crank_n is 1
    backw_e = 0   # sort of a magic number that will activate backward euler matrix if 1
    crank_n = 0   # same concept but for crank nickolson (will activate when -1)
    total_matrices = 1
    if discretization == 'Crank Nicholson':
        crank_n += 1
        backw_e += 1  # first matrix will have form of backw eul
        time = t[time_index] + deltat/2  # discretized at t+1/2*deltat
        total_matrices += 1
    elif discretization == 'Backward Euler':
        backw_e +=1
        time = t[time_index] + deltat  # discretized at t+deltat
    elif discretization == 'Forward Euler':
        time = t[time_index]  # discretized at t

    # if crank nicholson then create two matrices
    # one with form forward euler and one with form backwards euler (with lambda values *1/2)
    # (via * by pow(2,-crank_n))
    pde_matrices = []
    for k in range(total_matrices):
        size = mx+1
        bounds = np.array([0, mx+1])

        diag_appendage = np.array([0.0,0.0])  # initialised for the possibility of a neumann boundary
        total_diags = 3  # will be 5 if boundary is periodic
        diags = np.array([None for i in range(5)])
        diag_offsets = np.array([1, 0, -1, mx-1, -mx+1])  # last offset values for periodic boundary condition
        
        if k == 1:  # second matrix will have -lambda/2 in outer diagonals
            backw_e -= 1

        for i, condition in enumerate(boundary_conditions):
            # loop is made to elegantly handle mixed boundary conditions
            if condition[0] == 'Dirichlet':
                # dirichlet lhs boundary makes lambda arrays range from, at least, 1
                # dirichlet rhs boundary makes lambda arrays range to, at most, mx
                bounds[i] += pow(-1,i)
                size -= 1
            elif condition[0] == 'Neumann':
                # when i = 0 LHS BOUND : will take kappa at x[0] - deltax/2 ,to add (later) upper_diag[0]
                # when i = 1 RHS BOUND : will take kappa at x[mx] + deltax/2 ,to add (later) lower_diag[-1]
                diag_appendage[i] =  pow(2,-crank_n)*pow(-1,backw_e)*c*kappa(time,x[-i] + deltax/2 * pow(-1,i+1))
            elif condition[0] == 'Periodic':
                # sets variables for periodic boundary conditions and terminates the loop
                size = mx
                bounds = [0, mx]

                diag_appendage = [0,0]
                diags[[-2,-1]] = pow(2,-crank_n)*pow(-1,backw_e)* np.array([c*kappa(time,x[0]-deltax/2), c*kappa(time,x[mx]-deltax/2)])
                total_diags = 5
                break

        # works for variable kappa in x and t
        middle_diag = np.array([1 + pow(2,-crank_n)*pow(-1,backw_e+1)*c*(kappa(time,x[i] +deltax/2) + kappa(time,x[i] -deltax/2)) for i in range(bounds[0],bounds[1])])
        upper_diag = np.array([pow(2,-crank_n)*pow(-1,backw_e)*c*kappa(time,x[i] +deltax/2) for i in range(bounds[0],bounds[1]-1)])
        upper_diag[0] += diag_appendage[0]
        lower_diag = np.array([pow(2,-crank_n)*pow(-1,backw_e)*c*kappa(time,x[i] -deltax/2) for i in range(bounds[0]+1,bounds[1])])
        lower_diag[-1] += diag_appendage[1]

        diags[:3] = upper_diag, middle_diag, lower_diag
        pde_sparse = sparse.diags(diags[:total_diags], diag_offsets[:total_diags])
        pde_matrices.append(pde_sparse)

    # return single matrix for forward and backward euler
    # return two matrices if crank nicholson
    if discretization in ['Forward Euler', 'Backward Euler']:
        pde_matrix = pde_matrices[0]
        return pde_matrix
    else:
        pde_matrix_A, pde_matrix_B = pde_matrices[0], pde_matrices[1]
        return pde_matrix_A, pde_matrix_B

