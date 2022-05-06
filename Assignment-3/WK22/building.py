import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_diffusionV1 import *
from utility import *
from math import pi

def test_forw_euler_diffusion_rhsFUNC_x():
    '''Tests the numerical solution of 
    u_t = D*u_xx + f(x)    for u(t=0, x) = sin(pi*x/L) and f(x) = sin(pi*x/L)
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

    u_fe = forw_eul_diffusion(*args)
    rel_error = mean_rel_error(u_fe[-1], anal_u)

     return np.isclose(rel_error, 1, rtol=rtol)

test_forw_euler_diffusion_rhsFUNC_x()
print('yay')