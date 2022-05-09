import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from diffusion_dependencies import *
from utility import *
from math import pi
from scipy import sparse
from scipy.sparse.linalg import spsolve
import inspect

# forw_euler_sparse.diagonal(k) will get kth diagonal
# (t, x, u_I, kappa, l_boundary=lambda t:0, r_boundary=lambda t:0, rhs_func=lambda t,x:0, 

# SORT OUT BOUNDARY ARGS
# can get functions to work with variable kapp in t by making a new function that is kappa but with extra arguement of t
def solve_diffusion(t, x, kappa, u_I, boundary_conditions=[('Dirichlet', lambda t : 0), ('Dirichlet', lambda t : 0)], 
    rhs_func=lambda t,x : 0, discretization='Crank Nicholson',
    u_I_args=tuple(), u_I_kwargs=dict(),
    kappa_args=tuple(), kappa_kwargs=dict(),
    l_boundary_args=tuple(), l_boundary_kwargs=dict(), r_boundary_args=tuple(), r_boundary_kwargs=dict(),
    func_args=tuple(), func_kwargs=dict()):
    
    # make sure everything supplied in the correct format
    boundary_funcs, boundary_types = [], []
    for condition in boundary_conditions:
        boundary_types.append(condition[0])
        boundary_funcs.append(condition[1])

    index_boundary_position = ['left', 'right']
    valid_boundary_conditions = ['Dirichlet', 'Neumann', 'Periodic']
    valid_discretizations = ['Crank Nicholson','Backward Euler','Forward Euler']

    # assert that boundary conditions are valid
    assert all(boundary_condition in valid_boundary_conditions for boundary_condition in boundary_types), \
        'One or more of the boundary conditions entered is invalid. Exiting...'
    # functions arent used if boundary condition is periodic therefore check bound funcs if not periodic
    if 'Periodic' not in boundary_types:  
        assert all(callable(func) for func in boundary_funcs), 'Boundary conditions must be given as a function of t.'
        assert len(boundary_conditions) == 2, 'Boundary conditions must be supplied for both ends of the domain, unless condition is periodic'
        boundary_funcs_arglist = [inspect.getfullargspec(boundary_func).args for boundary_func in boundary_funcs]
        for arglist in boundary_funcs_arglist:
            assert arglist[0] == 't', 'Functions defining the boundary conditions must take t as input first.'

    # assert other functions are valid
    assert (callable(kappa)), 'Kappa must be given as a function of (t,x), or of (x,) if time invariant'
    assert (callable(u_I)), 'Inital distribution of u must be given as a function of (t,x)'
    assert (callable(rhs_func)), 'Right hand side function must be given as a function... duh'

    # assert that args are in the correct order
    kappa_arglist = inspect.getfullargspec(kappa).args
    uI_arglist = inspect.getfullargspec(u_I).args
    rhs_func_arglist =inspect.getfullargspec(rhs_func).args
    # does make the code less flexible but (almost) guarantees that incorrect input shouldnt be a problem
    assert kappa_arglist[:2] == ['t', 'x'] or (kappa_arglist[0] == 'x' and 't' not in kappa_arglist), \
        'Kappa function must take first positional arguements of (t,x), unless invariant in t. If invariant in t must take (x,) arguement first.'
    assert uI_arglist[0] == 'x','Initial distribution of u must take positional input (x) first.'
    assert rhs_func_arglist[:2] == ['t','x'],'Right hand side function must take positional inputs of (t,x) first, in that order.'

    # default to crank nicholson for invalid discretizations
    if discretization not in valid_discretizations:
        print('Discretization not recognized... Defaulting to Crank Nicholson.')
        discretization = 'Crank Nicholson'

    # checks complete
    # begin setting up algorithm
    
    u = np.zeros((t.size, x.size))        # initialise solution of pde
    u[0] = u_I(x, *u_I_args)

    # confirm boundary and initial condition are consistent
    # and set up bounds
    bounds = [0, x.size]
    for i, boundary_cond in enumerate(boundary_types):
        if boundary_cond == 'Dirichlet':
            # -0 indexes first element, -1 indexes last
            assert np.isclose(boundary_funcs[i](t[0],*l_boundary_args,**l_boundary_kwargs), u[0,-i]), \
                f'{index_boundary_position[i]} boundary condition is inconsistent with initial distribution of u.'
            # shorten bounds if dirichlet condition
            bounds[i] += pow(-1,i)
            # pre initialise boundaries if dirichlet
            u[:,-i] = boundary_funcs[i](t,*l_boundary_args,**l_boundary_kwargs)
    
    # since all my functions require kappa to take an arguement of t define new function with arguement of t
    # that will return the same value as kappa
    if 't' in kappa_arglist:
        kappa_t_dep = True
        def kappa_t(t,x,kappa_args,kappa_kwargs, kappa=kappa):
            D = kappa(t, x, *kappa_args, **kappa_kwargs) 
            return D
    else:
        kappa_t_dep = False
        def kappa_t(t,x,kappa_args,kappa_kwargs, kappa=kappa):
            D = kappa(x, *kappa_args, **kappa_kwargs) 
            return D

    # Due to the location of discretization backward euler and crank nicholson will not work with F (rhs func) dep. on u
    assert not (discretization in ['Backward Euler', 'Crank Nicholson'] and 'u' in rhs_func_arglist), \
        'Cannot solve for rhs function dependant on u with Crank Nicholson or Backward Euler discretizations.'
    
    match discretization:
        case 'Forward Euler':
            if 'u' in rhs_func_arglist:
                assert rhs_func_arglist[2] == 'u', 'u must be enetered as the third positional arguement for a rhs function dependant on u.'
                def rhs_func_u(t, x, u, args, kwargs, rhs_func=rhs_func):
                    f = rhs_func(t, x, u, *args, **kwargs)
                    return f
            else:
                # define F that is arbitrarily dependent on u
                def rhs_func_u(t, x, u, args, kwargs, rhs_func=rhs_func):
                    f = rhs_func(t, x, *args, **kwargs) 
                    return f

            u = forw_eul_solve_diffusion(u,t,x,kappa_t,kappa_t_dep,boundary_conditions,bounds,rhs_func_u,
                kappa_args, kappa_kwargs,func_args, func_kwargs,)
        case 'Backward Euler':
            u = backw_eul_solve_diffusion
        case 'Crank Nicholson':
            u = crank_nicholson_solve_diffusion
    
    return u

'''(t, x, kappa, u_I, boundary_conditions=[('Dirichlet', lambda t,x,u : 0), ('Dirichlet', lambda t,x,u : 0)], 
    rhs_func=lambda t,x,u : 0, discretization='Crank Nicholson',
    u_I_args=tuple(), u_I_kwargs=dict(),
    kappa_args=tuple(), kappa_kwargs=dict(),
    l_boundary_args=tuple(), l_boundary_kwargs=dict(), r_boundary_args=tuple(), r_boundary_kwargs=dict(),
    func_args=tuple(), func_kwargs=dict()):'''

def forw_eul_solve_diffusion(u,t,x,kappa,kappa_t_dep,boundary_conditions, bounds, rhs_func,
    kappa_args,kappa_kwargs,func_args,func_kwargs):

    
    deltat, deltax = get_grid_spacing(t,x)            
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)
    # define domain of x we are intersted in applying operations in
    x_domain = x[bounds[0]:bounds[1]]

    if kappa_t_dep:
        # will not check stability criteria since would have to iterate over a 2 dimensional space (V. LONG)
        # must generate matrix at each timestep since dependant on t
        for j in range(0, mt):
            u_j = u[j,bounds[0]:bounds[1]]
            forw_eul_mat = get_pde_solving_matrix(t,x,kappa,boundary_conditions,j,'Forward Euler',kappa_args,kappa_kwargs)
            cond_vector = get_condition_vector(t,x,kappa, boundary_conditions,j,'Forward Euler',kappa_args,kappa_kwargs)

            # apply matrix
            u_jp1 = forw_eul_mat.dot(u_j) 
            # apply rhs function
            u_jp1 += deltat*rhs_func(t[j],x_domain,u_j,func_args, func_kwargs)
            # apply cond vector
            u_jp1[[1,-1]] += cond_vector

            u[j+1,bounds[0]:bounds[1]] = u_jp1
    else:
        xx = np.linspace(x[0],x[-1],250)  # will be used to analyse stability criteria
        kappa_star = 1/2 * deltax**2/deltat

        kappa_xx = kappa(t, xx, kappa_args, kappa_kwargs)
        # check stability criteria 
        assert np.all(kappa_xx > 0), 'Stability criteria of forward Euler unsatisfied. Solutions will be inaccurate.\nExiting...'

        if np.any(kappa_xx < kappa_star):
            print('WARNING : stability criterion of forward euler may not be met, and solutions could possibly be inaccurate.\nContinuing...')
        
        # matrix not t dependent therefore time index parameter is not of importance
        forw_eul_mat = get_pde_solving_matrix(t,x,kappa,boundary_conditions,0,'Forward Euler')
        for j in range(0, mt):
            u_j = u[j,bounds[0]:bounds[1]]
            cond_vector = get_condition_vector(t,x,kappa, boundary_conditions,j,'Forward Euler')

            # apply matrix
            u_jp1 = forw_eul_mat.dot(u_j) 
            # apply rhs function
            u_jp1 += deltat*rhs_func(t[j],x_domain,u_j,func_args, func_kwargs)
            # apply cond vector
            u_jp1[[1,-1]] += cond_vector

            u[j+1,bounds[0]:bounds[1]] = u_jp1
    
    return u


def backw_eul_solve_diffusion(u,t,x,kappa,kappa_t_dep,boundary_conditions, bounds, rhs_func,
    kappa_args,kappa_kwargs,func_args,func_kwargs):

    
    deltat, deltax = get_grid_spacing(t,x)            
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)
    # define domain of x we are intersted in applying operations in
    x_domain = x[bounds[0]:bounds[1]]

    # No need to check for stability with backward euler
    if kappa_t_dep:
        for j in range(0, mt):
            u_j = u[j,bounds[0]:bounds[1]]
            backw_eul_mat = get_pde_solving_matrix(t,x,kappa,boundary_conditions,j,'Backward Euler')
            cond_vector = get_condition_vector(t,x,kappa, boundary_conditions,j,'Backward Euler')
            # must solve linear sys of equations:
            # A * u[j+1] = u[j] + cond.vector + deltat*rhs_func[j+1]

            b = u_j + deltat*rhs_func(t[j]+deltat,x_domain,*func_args, **func_kwargs)
            b[[1,-1]] += cond_vector
            # now in form Ax=b
            u_jp1 = spsolve(backw_eul_mat, b)

            u[j+1,bounds[0]:bounds[1]] = u_jp1
    else:
        
        # matrix not t dependent therefore time index parameter is not of importance
        backw_eul_mat = get_pde_solving_matrix(t,x,kappa,boundary_conditions,0,'Backward Euler')
        for j in range(0, mt):
            u_j = u[j,bounds[0]:bounds[1]]
            backw_eul_mat = get_pde_solving_matrix(t,x,kappa,boundary_conditions,j,'Backward Euler')
            cond_vector = get_condition_vector(t,x,kappa, boundary_conditions,j,'Backward Euler')
            # must solve linear sys of equations:
            # A * u[j+1] = u[j] + cond.vector + deltat*rhs_func[j+1]

            b = u_j + deltat*rhs_func(t[j]+deltat,x_domain,*func_args, **func_kwargs)
            b[[1,-1]] += cond_vector
            # now in form Ax=b
            u_jp1 = spsolve(backw_eul_mat, b)

            u[j+1,bounds[0]:bounds[1]] = u_jp1
    
    return u

def crank_nicholson_solve_diffusion(u,t,x,kappa,kappa_t_dep,boundary_conditions, bounds, rhs_func,
    kappa_args,kappa_kwargs,func_args,func_kwargs):

    
    deltat, deltax = get_grid_spacing(t,x)            
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)
    # define domain of x we are intersted in applying operations in
    x_domain = x[bounds[0]:bounds[1]]

    # No need to check for stability with crank nicholson
    if kappa_t_dep:
        for j in range(0, mt):
            u_j = u[j,bounds[0]:bounds[1]]
            crank_nicholson_mat_A, crank_nicholson_mat_B = get_pde_solving_matrix(t,x,kappa,boundary_conditions,j,'Crank Nicholson')
            cond_vector = get_condition_vector(t,x,kappa,boundary_conditions,j,'Crank Nicholson')
            # must solve linear sys of equations:
            # A * u[j+1] = B * u[j] + cond.vector + deltat*rhs_func[j+1/2]

            b = crank_nicholson_mat_B.dot(u_j) + deltat*rhs_func(t[j]+deltat/2,x_domain,*func_args, **func_kwargs)
            b[[1,-1]] += cond_vector
            # now in form Ax=b
            u_jp1 = spsolve(crank_nicholson_mat_A, b)

            u[j+1,bounds[0]:bounds[1]] = u_jp1

    else:
        # matrix not t dependent therefore time index parameter is not of importance
        backw_eul_mat = get_pde_solving_matrix(t,x,kappa,boundary_conditions,0,'Crank Nicholson')
        for j in range(0, mt):
            u_j = u[j,bounds[0]:bounds[1]]
            cond_vector = get_condition_vector(t,x,kappa, boundary_conditions,j,'Crank Nicholson')
            # must solve linear sys of equations:
            # A * u[j+1] = B * u[j] + cond.vector + deltat*rhs_func[j+1/2]

            b = crank_nicholson_mat_B.dot(u_j) + deltat*rhs_func(t[j]+deltat/2,x_domain,*func_args, **func_kwargs)
            b[[1,-1]] += cond_vector
            # now in form Ax=b
            u_jp1 = spsolve(crank_nicholson_mat_A, b)

            u[j+1,bounds[0]:bounds[1]] = u_jp1
    
    return u