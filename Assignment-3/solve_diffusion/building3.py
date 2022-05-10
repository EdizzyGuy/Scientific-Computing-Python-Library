import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_diffusionV3 import *
from utility import *
from math import pi
from scipy import sparse
from scipy.sparse.linalg import spsolve
import inspect
import time

pde_matrices = []
        pde_sparse = sparse.diags(diags[:total_diags], offset[:total_diags])
        pde_matrices.append(pde_sparse)
def get_pde_solving_matrix(t,x,kappa,boundary_conditions=[('Dirichlet',), ('Dirichlet',)], time_index=0, discretization='Crank Nicholson',
    kappa_args=tuple(), kappa_kwargs=dict()):

    matrix_diags = get_pde_solving_diags(t,x,kappa,boundary_conditions, time_index, discretization, kappa_args, kappa_kwargs)
    Matrices = []
    shape = (len(matrix_diags[0,]))
    for diagonal_matrix in matrix_diags:
        Matrix = np.array((len()))
        for diags, offsets in diagonal_matrix:
            np.diag(diag)




   
def get_pde_solving_diags(t,x,kappa,boundary_conditions=[('Dirichlet',), ('Dirichlet',)], time_index=0, discretization='Crank Nicholson',
    kappa_args=tuple(), kappa_kwargs=dict()):
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
        t (np.ndarray)                          : discretized values in time of the grid space.
        x (np.ndarray)                          : discretized values of x in the grid space.
        boundary_conditions (list of tuple)     : list of classifications of the boundary conditions on the 1D domain. Boundary conditions given as a tuple like so:
                                                (condition type (str), function of condition (callable)). First entry is the boundary condition on the left side of 
                                                the domain, and second entry is the boundary condition on the right side of the domain.
        time_index (int)                        : value of j such that current time = t[j]. Used in the case that kappa is variable in t.
        discretization (str)                    : The kind of discretization used to solve the pde. This will affect the outputted matrix.
        kappa_args (tuple)                      : Tuple of any additional positional arguements to be passed to kappa.
        kappa_kwargs (dict)                     : Dictionary of key word arguements to be passed to kappa.

    Returns: 
        pde_diags_A (np.ndarray)  : Diagonals of 1st matrix required
        pde_diags_B (np.ndarray)  : Diagonals of second matrix required, if applicable.
    '''
    
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

    
    total_diags = 3  # will be 5 if boundary is periodic
    offset = np.array([-1, 0, 1, mx-1, -mx+1])  # last offset values for periodic boundary condition
    diags = np.array([None for i in range(5)] for j in range(total_matrices))  # 2d array: 1st indx: matrix in question, 2nd: each diag

    # if crank nicholson then create two matrices
    # one with form forward euler and one with form backwards euler (with lambda values *1/2)
    # (via * by pow(2,-crank_n))
    for k in range(total_matrices):
        size = mx+1
        bounds = np.array([0, mx+1])

        diag_appendage = np.array([0.0,0.0])  # initialised for the possibility of a neumann boundary
        
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
                diag_appendage[i] =  pow(2,-crank_n)*pow(-1,backw_e)*c*kappa(time,x[-i] + deltax/2 * pow(-1,i+1), kappa_args, kappa_kwargs)
            elif condition[0] == 'Periodic':
                # sets variables for periodic boundary conditions and terminates the loop
                size = mx
                bounds = [0, mx]

                diag_appendage = [0,0]
                diags[[-2,-1]] = pow(2,-crank_n)*pow(-1,backw_e)* np.array([c*kappa(time,x[0]-deltax/2,kappa_args,kappa_kwargs), c*kappa(time,x[mx]-deltax/2,kappa_args,kappa_kwargs)])
                total_diags = 5
                break

        # works for variable kappa in x and t
        middle_diag = np.array([1 + pow(2,-crank_n)*pow(-1,backw_e+1)*c*(kappa(time,x[i] +deltax/2,kappa_args,kappa_kwargs) + kappa(time,x[i] -deltax/2,kappa_args,kappa_kwargs)) for i in range(bounds[0],bounds[1])])
        upper_diag = np.array([pow(2,-crank_n)*pow(-1,backw_e)*c*kappa(time,x[i] +deltax/2,kappa_args,kappa_kwargs) for i in range(bounds[0],bounds[1]-1)])
        upper_diag[0] += diag_appendage[0]
        lower_diag = np.array([pow(2,-crank_n)*pow(-1,backw_e)*c*kappa(time,x[i] -deltax/2,kappa_args,kappa_kwargs) for i in range(bounds[0]+1,bounds[1])])
        lower_diag[-1] += diag_appendage[1]

        diags[k,:3] = lower_diag, middle_diag, upper_diag

    diagonals = diags[:,:total_diags]  # crank nicolson matrices are symmmetric
    return diagonals, offset



        
        