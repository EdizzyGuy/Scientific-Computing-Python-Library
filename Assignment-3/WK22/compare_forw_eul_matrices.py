#%%
from solve_heat_eq3 import *
from solve_diffusionV1 import *
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

'''
Average time for generating matrix with old method (with wrapper func) : 9.53E-05
Standard deviation                                                     : 8.83E-04
Average time for generating matrix with variable kappa in x            : 5.76E-04
Standard deviation   
Average time for generating matrix with variable kappa in x and t      : 2.92E+00
Standard deviation                                                     : 4.33E-02  
'''
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

def kappa_func_x(x):
    kappa = 1.0
    return kappa

def dummy_wrapper_euler_forw(t,x,kappa, var_diffusion=False):
    match var_diffusion:
        case True:
            pde_matrix = forw_eul_pde_matrix_varKappa_tx(t, x, kappa)
        case False:
            deltat, deltax = get_grid_spacing(t,x)
            lmbda = kappa*deltat/(deltax**2)
            pde_matrix = forw_eul_pde_matrix(lmbda, mx)

    return pde_matrix
#%%
count=1000
avg_wrapper, std_wrapper = time_function(dummy_wrapper_euler_forw, count, (t,x,kappa,False))
#avg_tx, std_tx = time_function(forw_eul_pde_matrix_varKappa_tx,count,(t,x,kappa_func))
avg_x, std_x = time_function(forw_eul_pde_matrix_varKappa_x,count,(t,x,kappa_func_x))

#%%
print(f'Average time for generating matrix with old method (with wrapper func) : {avg_wrapper:.2E}\n'
    f'Standard deviation                                                     : {std_wrapper:.2E}\n\n')

print(f'Average time for generating matrix with variable kappa in x            : {avg_x:.2E}\n'
      f'Standard deviation                                                     : {std_x:.2E}\n\n')

#print(f'Average time for generating matrix with variable kappa in x and t      : {avg_tx:.2E}\n'
#      f'Standard deviation                                                     : {std_tx:.2E}\n\n')


# %%
count = 1000
avg_new_x, std_new_x = time_function(get_forward_euler_matrix_x,count,(t,x,kappa_func_x))

print(f'Average time for np.diag with variable kappa in x                      : {avg_new_x:.2E}\n'
      f'Standard deviation                                                     : {std_new_x:.2E}\n\n')

# %%
