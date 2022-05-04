#%%
from solve_heat_eq3 import *
from solve_diffusionV1 import *
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

mx=100
mt=10000
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1) 
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]  

def kappa_func(t,x):
    kappa = 1.0
    return kappa

def dummy_wrapper_euler_forw(kappa, var_diffusion=False):
    match var_diffusion:
        case True:
            pde_matrix = forw_eul_pde_matrix_varKappa_tx(t, x, kappa)
        case False:
            lmbda = kappa*deltat/(deltax**2)
            pde_matrix = forw_eul_pde_matrix(lmbda, mx)

    return pde_matrix
#%%
count=1
avg_wrappper, std_wrapper = time_function(dummy_wrapper_euler_forw, count, (kappa,))
avg, std = time_function(forw_eul_pde_matrix_varKappa_tx,count,(t,x,kappa_func))

# %%
