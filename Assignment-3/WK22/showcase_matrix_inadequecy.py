import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_diffusionV2 import *
from utility import *
from math import pi

'''
        for i in range(1, mx):
            u[j+1,i] = u[j,i] + lmbda*(u[j,i-1] - 2*u[j,i] + u[j,i+1]) + deltat*rhs_func(t[j],x[i])  # find solution forward 1 step in time
        u[j+1,0] = l_boundary(t[j+1])
        u[j+1,-1] = r_boundary(t[j+1])
        test = u_mat[j+1] == u[j+1]
        '''

def forw_eul_diffusion1(t, x, u_I, kappa, l_boundary=lambda t:0, r_boundary=lambda t:0, rhs_func=lambda t,x:0, 
        u_I_args=tuple(), kappa_args=tuple(), kappa_kwargs=dict(), func_args=tuple()):
    # only works for 0 boundary conditions
        
    deltat, deltax = get_grid_spacing(t,x)            
    mt, mx = int(t[-1] / deltat), int(x[-1] / deltax)
    
    u = np.zeros((t.size, x.size))        # initialise solution of pde
    u_mat = np.zeros((t.size, x.size))  
    for i in range(0, mx+1):
        u[0,i] = u_I(x[i], *u_I_args)
    u_mat[0] = u_I(x, *u_I_args)  # FASTER THIS WAY
    print(np.all(u[0] == u_mat[0]))
    # confirm boundary and initial condition are consistent
    assert np.isclose(u[0,0], l_boundary(t[0])) and np.isclose(u[0,-1], r_boundary(t[0]))
    # initialise boundary values
    u[:,0] = l_boundary(t)
    u[:,-1] = r_boundary(t)
    

    if callable(kappa):
        args = inspect.getfullargspec(kappa).args
        xx = np.linspace(x[0],x[-1],250)  # will be used to analyse stability criteria 
        kappa_star = 1/2 * deltax**2/deltat

        if 't' in args:
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

    for j in range(mt):
        m1 = np.matmul(forw_eul_matrix[j*oracle],u_mat[j,1:-1]) 
        n2 = np.zeros(x.size - 2)
        m2 = deltat*rhs_func(t[j],x[1:-1],*func_args) # vector of additives from rhs func

        # vector of u between boundaries
        n1 = np.zeros(x.size - 2)

        # below will implement matrix multiplication step wise
        for i in range(0, mx-1):
            # by default elements next to the boundary will recieve
            # information from the boundary
            n1[i] = lmbda*u[j,i] + (1-2*lmbda)*u[j,i+1] + lmbda*u[j,i+2] 
            n2[i] = deltat*rhs_func(t[j],x[i+1])  # find solution forward 1 step in time

        print(np.all(m1[1:-1] == n1[1:-1])) 
        # mid values of m1 and n1 are the same 
        # (just need to make sure the appendage onto ends make them equal)
        boundary_j = np.array([l_boundary(t[j]), r_boundary(t[j])])
        bound_appendage_m = np.multiply(boundary_j,bound_lambda[j*oracle])
        m1[[0,-1]] += bound_appendage_m

        test = np.all(n1==m1) and np.all(n2==m2)
        print(test)

        u[j+1,1:-1] = n1+n2
        u_mat[j+1,1:-1] = m1+m2 

        
        u_mat[j+1,0] = l_boundary(t[j+1])
        u_mat[j+1,-1] = r_boundary(t[j+1])
    return u, u_mat

def test_forw_euler_diffusion_nohomo_dirich():
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
        y = 3*np.sin(2*pi*x) + 2*(1-x)
        return y
    def rhs_func(t,x):
        f = x
        return f
    def u_exact(x,t):
        # the exact solution
        y = 3*np.sin(2*pi*x)*np.exp(-4*pi**2*t) +2 + (t-2)*x
        return y
    #u(0,t) = 2; u(L,t) = t
    def l_bound(t):
        return 2
    def r_bound(t):
        return t
    
    mx = 100
    mt = 10000

    t, x = get_grid_space(T,L,mt,mx)     # mesh points in space and time

    anal_u = u_exact(x,T)
    real_u = np.zeros((t.size, x.size))
    for i, j in enumerate(t):
        real_u[i] = u_exact(x,j)

    args = (t,x,u_I,kappa)
    kwargs = {'l_boundary':l_bound,'r_boundary':r_bound,'rhs_func': rhs_func}

    u_mat = forw_eul_diffusion_use_matrix(*args, **kwargs)
    u_fe, u_mat = forw_eul_diffusion1(*args, **kwargs)
    rel_error = mean_rel_error(u_fe[-1], anal_u)
    rel_error_mat = mean_rel_error(u_mat[-1], anal_u)

    return u_fe, u_mat, x, t, real_u, rel_error, rel_error_mat

u, u_mat, x, t, real_u, rel_error, rel_error_mat = test_forw_euler_diffusion_nohomo_dirich()
print('A relative error of 1 means there is no error. \n')
print(f'Relative accuracy of solution using step wise forward euler : {rel_error}')
print(f'Relative accuracy of solution using step wise forward euler : {rel_error_mat}\n')
print('displaying true solution...')
display_dynamic_solution(real_u,x,t,'True Solution')
print('\ndisplaying matrix calculated solution...')
display_dynamic_solution(u_mat,x,t, 'Matrix calculated Solution')
print('\ndisplaying step wise calculated solution...')
display_dynamic_solution(u,x,t, 'stepwise calculated')



print('yay')