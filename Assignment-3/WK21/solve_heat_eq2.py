import numpy as np

'''
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

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


def forw_eul_pde_matrix(lmbda, mx):
    # MATRIX IMPLEMENTATION
    lambda_array = np.array([lmbda for i in range(mx - 1)])
    A_FE_1 = np.diag(1 - 2* lambda_array)
    A_FE_2 = np.diag(lambda_array[:-1], k=1)
    A_FE_3 = np.diag(lambda_array[:-1], k=-1)
    A_FE = A_FE_1 + A_FE_2 + A_FE_3

    return A_FE


def forw_eul_pde_step(u_j, lmbda, mx):
        # Solve the PDE: loop over all time points
    # Forward Euler timestep at inner mesh points
    # PDE discretised at position x[i], time t[j]
    u_jp1 = np.zeros(u_j.shape)  # boundary condition set
    for i in range(1, mx):
        u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])  # find solution forward 1 step in time

    return u_jp1


def forw_eul_heat_eq(t, x, u_I, kappa, method='Matrix', u_I_args=tuple()):
    # only works for 0 boundary conditions
    
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t

    mx = int(x[-1] / deltax)
    mt = int(t[-1] / deltat)
    lmbda = kappa*deltat/(deltax**2) 
    if lmbda > 0.5:
        print('Stability criteria not met, solutions to the heat equation will be inaccurate') 
    
    u = np.zeros((t.size, x.size))        # initialise solution of pde
    # first index is time

    for i in range(0, mx+1):
        u[0,i] = u_I(x[i], *u_I_args)
    
    match method:
        case 'Singular': 
            for j in range(0,mt):
                u_j = u[j]
                u_jp1 = forw_eul_pde_step(u_j, lmbda, mx)
                u[j+1] = u_jp1
        case 'Matrix':
            pde_matrix = forw_eul_pde_matrix(lmbda, mx)
            for j in range(0, mt):
                u[j+1,1:-1] = pde_matrix @ u[j,1:-1]
                u[j+1,[0,-1]] = 0  # boundary conditions

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

    
