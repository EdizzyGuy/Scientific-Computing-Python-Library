import numpy as np
from scipy import linalg
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
    mt = kappa * T * mx**2 /(max_lambda * L**2)
    return int(np.floor(mt))

def find_suitable_mx(mt, kappa, L, T, max_lambda=0.5):
    mx = np.sqrt(max_lambda * L**2 * mt /(kappa * T))
    return int(np.floor(mx))

def get_grid_space(T, L, mt, mx):
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time           
    return t, x

def get_grid_spacing(t,x): 
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]
    return deltat, deltax

def forw_eul_pde_matrix(lmbda, mx, boundary='Dirichlet'):
    ''' Gets required matrix for using forward euler sicretisation to approximate the diffusion eq. for a CONSTANT diffusion constant
    i.e. such that u[j+1] = forw_eul_matrix @ u[j]

    Args:
        lmbda (float) : value of lambda for values of kappa, deltat and deltax
        mx (int)      : number of gridpoints in space
        boundary (str): type of boundary condition in question

    Returns:
        A_FE          : matrix A_FE such that u[j+1] = A_FE @ u[j] for forward euler discretisation 
    '''
    # MATRIX IMPLEMENTATION
    match boundary:
        case 'Dirichlet':
            size = mx-1
            additive = [0,0]
        case 'Neumann':
            size = mx+1
            additive = [lmbda, lmbda]

    lambda_array = np.array([lmbda for i in range(size)])
    # diag entries (lmbda, 1-2*lmbda, lmbda)
    A_FE_1, A_FE_2, A_FE_3 = np.diag(1 - 2* lambda_array), np.diag(lambda_array[:-1], k=1), np.diag(lambda_array[:-1], k=-1)
    A_FE = A_FE_1 + A_FE_2 + A_FE_3

    A_FE[[0,-1],[1,-2]] += additive
    A_FE = A_FE[np.newaxis, ...]  # change dimensionality of A_FE so that code can work in generality
    return A_FE

#TODO implement this using np.diag and compare speed
def forw_eul_pde_matrix_varKappa_x(t, x, kappa, boundary='Dirichlet', args=tuple(), kwargs=dict()):
    ''' Gets required matrix for using forward euler sicretisation to approximate the diffusion eq. for a diffusion coefficient that varies in x.
    i.e. such that u[j+1] = forw_eul_matrix @ u[j].

    Args:
        t (np.ndarray)   : gridpoints in time
        x (n.ndarray)    : gridpoints in first spatial dimension
        kappa (callable) : Function that defines value of kappa within the space, and takes arguements (x, *args)
        args (tuple)     : additional positional arguements to be passed to kappa.
        kwargs (dict)    : keyword arguements to be passed to kappa

    Returns:
        A                : matrix A such that u[j+1] = A @ u[j] for forward euler discretisation and 0 dirichlet bound conditions

    '''
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    
    c = deltat / deltax**2
    mx = int(x[-1] / deltax)

    match boundary:
        case 'Dirichlet':
            size = mx-1
            additive = [0,0]
            x = x[1:-1]  # modify x matrix so that inputted i in the for loop will be discretized about the correct values, although :-1 isnt strictly neccessary 

        case 'Neumann':
            size = mx+1
            additive = [c*kappa(x[0] + deltax/2), c*kappa(x[-1] - deltax/2)]

    A = np.zeros((1,size,size))  # create 3d array so that all forw eul mat have the same dimensionality
    # tridiag   (c*k_{i-1/2},   1- c*(k_{i+1/2} + k_{i-1/2}, c*k_{i+1/2}))
    for i in range(size):
        A[0,i,i-1] = c*kappa(x[i]-deltax/2,*args,**kwargs)
        A[0,i,i] = 1 - c*(kappa(x[i]+deltax/2,*args) + kappa(x[i]-deltax/2,*args,**kwargs))
        A[0,i,(i+1)%(mx-1)] = c*kappa(x[i+1]+deltax/2,*args,**kwargs)

    A[0,0,-1] = 0
    A[0,-1,0] = 0
    A[[0,-1],[1,-2]] += additive

    return A

#TODO implement this using np.diag and compare speed
def forw_eul_pde_matrix_varKappa_tx(t, x, kappa, args=tuple()):
    ''' Gets required matrix for using forward euler sicretisation to approximate the diffusion eq. for a diffusion coefficient that varies in x.
    i.e. such that u[j+1] = forw_eul_matrix @ u[j].
        NOTE: This function assumes kappa is a function of t and x, therefore outputted array will be {gridpoints in time} times greater than 
    the standard forward euler matrix. This makes it slow for large number of gridpoints in time.

    Args:
        t (np.ndarray)   : gridpoints in time
        x (n.ndarray)    : gridpoints in first spatial dimension
        kappa (callable) : Function that defines value of kappa within the space, and takes arguements (t, x, *args)
        args (tuple)     : additional arguements to be passed to kappa.

    Returns:
        A                : matrix A such that u[j+1] = A @ u[j] for forward euler discretisation 

    '''
    
    deltat, deltax = get_grid_spacing(t,x)           # gridspacing in t
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)

    c = deltat / deltax**2
    A = np.zeros((mt, mx-1, mx-1))
    for j in range(mt):
        for i in range(mx-1):
            A[j,i,i-1] = c*kappa(t[j],x[i-1]-deltax/2,*args)
            A[j,i,i] = 1 - c*(kappa(t[j],x[i]+deltax/2,*args) + kappa(t[j],x[i]-deltax/2,*args))
            A[j,i,(i+1)%(mx-1)] = c*kappa(t[j],x[i+1]+deltax/2,*args)
        A[j,0,-1] = 0
        A[j,-1,0] = 0
    
    return A

def forw_eul_diffusion_boundary(t,x,kappa,t_dep=False,boundary='Dirichlet',args=tuple(),kwargs=dict()):
    """This function will return, as an array, the value of lambda at the discretizations of forward euler for the discretization
    that requires information from the boundary condition in order to solve for u[j+1]. This function is required for forward Euler
    to solve diffusion equations with boundary conditions that are not homogenous dirichlet, and have a variable kappa. 
    In the case of Dirichlet this is the discretization at point 1 and mx-1,
    In the case of Neumann this is the discretization at point 0 and mx

    Args:
        t (np.ndarray)              : Array of discretized t values.
        x (np.ndarray)              : Array of discretized x values.
        kappa (callable)            : Function for diffusion coefficient
        boundary (str)              : Boundary type in question
        args (tuple)                : Positional arguements to be passed to kappa
        kwargs (dict)               : Keyword arguements to be passed to kappa

    Returns:
        boundr_lambda (np.ndarray)  : Value of lambda at the boundary of the discretized domain
    """
    valid_boundaries = ['Dirichlet', 'Neumann']
    assert boundary in valid_boundaries, 'Boundary type inputted is invalid. Exiting...'


    match boundary:
        case 'Dirichlet':
            discretizations = [1, -2]
        case 'Neumann':
            discretizations = [0, -1]

    oracle = 0
    if t_dep == True:
        oracle = 1
        Kappa = kappa
    else:
        oracle = 0
        # artificial t dependency so that two functions need not be made
        Kappa = lambda t,x,*args : kappa(x, *args, **kwargs)

    deltat, deltax = get_grid_spacing(t,x)
    mt = len(t) -1

    boundr_lambda = np.zeros((oracle*(mt-1) +1,2))
    for j in range(oracle*(mt-1) +1):  # guaranteed to be 1 or mt
        # left : k_i-1/2
        # right : k_i+1/2
        boundr_lambda[j,0] = Kappa(t[j],x[discretizations[0]] -deltax/2)* deltat / deltax**2
        boundr_lambda[j,-1] = Kappa(t[j],x[discretizations[1]] +deltax/2)* deltat / deltax**2
    
    return boundr_lambda


def forw_eul_pde_step_constKappa(u_j, lmbda, mx):
        # Solve the PDE: loop over all time points
    # Forward Euler timestep at inner mesh points
    # PDE discretised at position x[i], time t[j]
    u_jp1 = np.zeros(u_j.shape)  # boundary condition set
    for i in range(1, mx):
        u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])  # find solution forward 1 step in time
    # do not return boundary values.
    return u_jp1[1:-1]

def forw_eul_diffusion_use_steps(t, x, u_I, kappa, l_boundary=lambda t:0, r_boundary=lambda t:0, rhs_func=lambda t,x:0, 
        u_I_args=tuple(), kappa_args=tuple(), func_args=tuple()):
        
    deltat, deltax = get_grid_spacing(t,x)  # distance between points on grid in x and t           
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)  # No of grid points in x and t
    
    u = np.zeros((t.size, x.size))        
    # initialise solution of pde
    for i in range(0, mx+1):
        u[0,i] = u_I(x[i], *u_I_args)
    # confirm boundary and initial condition are consistent
    assert np.isclose(u[0,0], l_boundary(t[0])) and np.isclose(u[0,-1], r_boundary(t[0]))
    # initialise boundary values
    u[:,0] = l_boundary(t)
    u[:,-1] = r_boundary(t)
    

    if callable(kappa):  # kappa is a fucntion
        xx = np.linspace(x[0],x[-1],250)  # will be used to analyse stability criteria 
        kappa_star = 1/2 * deltax**2/deltat 

        # check stability criterion
        kappa_xx = kappa(xx, *kappa_args)
        assert np.all(kappa_xx > 0), 'Stability criteria of forward Euler unsatisfied. Solutions will be inaccurate.\nExiting...'
        if np.any(kappa_xx < kappa_star):
            print('WARNING : stability criterion of forward euler may not be met, and solutions could possibly be inaccurate.\nContinuing...')
        
    else:  # constant kappa
        # check if stability criteria is met
        lmbda = kappa*deltat/(deltax**2) 
        assert lmbda <= 0.5, 'Stability criteria of forward Euler unsatisfied. Solutions will be inaccurate. Exiting...'
        for j in range(0,mt):
            u[j+1,1:-1] = forw_eul_pde_step_constKappa(u[j], lmbda, mx)
    
    return u

def forw_eul_diffusion_use_matrix(t, x, u_I, kappa, l_boundary=lambda t:0, r_boundary=lambda t:0, rhs_func=lambda t,x:0, 
        u_I_args=tuple(), u_I_kwargs=dict(),
        kappa_args=tuple(), kappa_kwargs=dict(),
        func_args=tuple(), func_kwargs=dict()):
        
    deltat, deltax = get_grid_spacing(t,x)            
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)
    
    u = np.zeros((t.size, x.size))        # initialise solution of pde
    u[0] = u_I(x, *u_I_args)
    
    # confirm boundary and initial condition are consistent
    assert np.isclose(u[0,0], l_boundary(t[0])) and np.isclose(u[0,-1], r_boundary(t[0]))
    # initialise boundary values
    u[:,0] = l_boundary(t)
    u[:,-1] = r_boundary(t)
    

    if callable(kappa):
        args = inspect.getfullargspec(kappa).args
        xx = np.linspace(x[0],x[-1],250)  # will be used to analyse stability criteria 
        kappa_star = 1/2 * deltax**2/deltat

        if any(['t', 'T',"time"]) in args:  #HAS KIERAN FCKED ME UP?
            # is not worth checking stability in case of t dependence since evaluated array would be massive!
            forw_eul_matrix = forw_eul_pde_matrix_varKappa_tx(t,x,kappa,*kappa_args, **kappa_kwargs)
            bound_lambda = forw_eul_diffusion_boundary(t,x,kappa,t_dep=True,args=kappa_args,kwargs=kappa_kwargs)
            # this variable will tell the program whether there will be a different A matrix at each time step
            # see below underneath the termination of the overhanging if statement
            oracle = 1  
        else:
            # check stability criterion
            kappa_xx = kappa(xx, *kappa_args, **kappa_kwargs)
            assert np.all(kappa_xx > 0), 'Stability criteria of forward Euler unsatisfied. Solutions will be inaccurate.\nExiting...'

            if np.any(kappa_xx < kappa_star):
                print('WARNING : stability criterion of forward euler may not be met, and solutions could possibly be inaccurate.\nContinuing...')

            forw_eul_matrix = forw_eul_pde_matrix_varKappa_x(t,x,kappa,*kappa_args, **kappa_kwargs)
            bound_lambda = forw_eul_diffusion_boundary(t,x,kappa,t_dep=False,args=kappa_args,kwargs=kappa_kwargs)
            oracle = 0
    else:  # constant kappa
        # check if stability criteria is met
        lmbda = kappa*deltat/(deltax**2) 
        assert lmbda <= 0.5, 'Stability criteria of forward Euler unsatisfied. Solutions will be inaccurate. Exiting...'

        forw_eul_matrix = forw_eul_pde_matrix(lmbda, mx)
        bound_lambda = np.array([[lmbda, lmbda]])
        oracle = 0

    for j in range(0, mt):
        #apply forward euler and rhs_function
        euler = forw_eul_matrix[ j*oracle] @ u[j,1:-1] 
        rhs = deltat*rhs_func(t[j],x[1:-1],*func_args, **func_kwargs) # the oracle shines
        # apply inhomogenous dirichlet boundary modifications to forward euler
        boundary_j = np.array([l_boundary(t[j]), r_boundary(t[j])])
        dirichlet_additive = np.multiply(boundary_j,bound_lambda[j*oracle])
        euler[[0,-1]] += dirichlet_additive
        
        u[j+1,1:-1] = euler + rhs
    return u

def back_eul_pde_matrix(lmbda, mx):
    lambda_array = np.array([lmbda for i in range(mx - 1)])
    A_BE_1 = np.diag(1 + 2* lambda_array)
    A_BE_2 = np.diag(-1 * lambda_array[:-1], k=1)
    A_BE_3 = np.diag(-1 * lambda_array[:-1], k=-1)
    A_BE = A_BE_1 + A_BE_2 + A_BE_3

    return A_BE


def back_eul_heat_eq(t, x, u_I, kappa, u_I_args=tuple()):
  
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t

    mx = int(x[-1] / deltax)
    mt = int(t[-1] / deltat)
    lmbda = kappa*deltat/(deltax**2) 

    u = np.zeros((t.size, x.size))        # initialise solution of pde
    # first index is time
    for i in range(0, mx+1):
        u[0,i] = u_I(x[i], *u_I_args)

    pde_matrix = back_eul_pde_matrix(lmbda, mx)
    mat_inv = np.linalg.inv(pde_matrix)
    # A_BE * u_j+1 = u_j
    # u_j+1 = inv(A_BE) * u_j
    for j in range(0, mt):
        u[j+1, 1:-1] = mat_inv @ u[j,1:-1]
        u[j+1,[0,-1]] = 0  # bound. cond.

    return u


def crank_nicholson_pde_matrix(lmbda, mx):
    lambda_array = np.array([lmbda for i in range(mx - 1)])
    # A_CN = tridiag(-lmbda/2, 1+lambda, -lambda/2)
    A_CN_1 = np.diag(1 + lambda_array)
    A_CN_2 = np.diag(-1/2 * lambda_array[:-1], k=1)
    A_CN_3 = np.diag(-1/2 * lambda_array[:-1], k=-1)
    A_CN = A_CN_1 + A_CN_2 + A_CN_3
    # B_CN = tridiag(lamdba/2, 1-lamdba, lamdba/2)
    B_CN_1 = np.diag(1 - lambda_array)
    B_CN_2 = np.diag(1/2 * lambda_array[:-1], k=1)
    B_CN_3 = np.diag(1/2 * lambda_array[:-1], k=-1)  # CAN JUST USE MATRIX TRANSPOSE OF B_CN_2
    B_CN = B_CN_1 + B_CN_2 + B_CN_3

    return A_CN, B_CN

    
def crank_nichol_eul_heat_eq(t, x, u_I, kappa, method='linalg solve', u_I_args=tuple()):

    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t

    mx = int(x[-1] / deltax)
    mt = int(t[-1] / deltat)
    lmbda = kappa*deltat/(deltax**2) 

    u = np.zeros((t.size, x.size))        # initialise solution of pde
    # first index is time
    for i in range(0, mx+1):
        u[0,i] = u_I(x[i], *u_I_args)

    A, B = crank_nicholson_pde_matrix(lmbda, mx)
    # A_CN * u_j+1 = B_CN * u_j
    # u_j+1 = inv(A_CN) * B_CN * u_j
    match method:
        case 'linalg solve':
            for j in range(0, mt):
                b = B @ u[j,1:-1]
                # solves a x = b for x
                u[j+1, 1:-1] = linalg.solve(A, b)
                u[j+1, [0,-1]] = 0  # bound. cond.
        case 'matrix inversion':
            C = np.linalg.inv(A) @ B
            for j in range(0, mt):
                u[j+1, 1:-1] = C @ u[j,1:-1]
                u[j+1, [0,-1]] = 0  # bound. cond.

    return u