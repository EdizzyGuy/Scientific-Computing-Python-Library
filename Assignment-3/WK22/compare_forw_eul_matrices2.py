import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_diffusionV1 import *
from utility import *
from math import pi

'''
duration of defining matrix for constant kappa (as a function):
    0.0067217350006103516
duration of defining matrix for constant kappa
    0.007536649703979492
'''

def test_forw_euler_diffusion():

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
    args2 = (t,x,u_I,1.0)

    s = time.time()
    u_fe = forw_eul_diffusion_use_matrix(*args)
    e = time.time()
    u_fe2 = forw_eul_diffusion_use_matrix(*args2)
    f = time.time()
    rel_error = mean_rel_error(u_fe[-1], anal_u)
    rel_error2 = mean_rel_error(u_fe2[-1], anal_u)

    return rel_error, rel_error2, e-s, f-e

rel_error = test_forw_euler_diffusion()

print('yay')