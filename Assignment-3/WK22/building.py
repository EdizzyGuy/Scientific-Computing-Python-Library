import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_diffusionV1 import *
from utility import *
from math import pi
from scipy import sparse

# forw_euler_sparse.diagonal(k) will get kth diagonal
#%%
def get_conditional_effects(t,x,kappa,boundary_class=['Dirichlet', 'Dirichlet'], boundary_condition=[lambda t,x: 0, lambda t,x: 0] time_index=0, discretization='Crank Nicholson'):
    # no need to do forward euler since you can just append any answers at the start
    # CAN YOU HAVE BOUNDARY CONDITION DEPENDANT ON U?

    deltat, deltax = get_grid_spacing(t,x)
    mt, mx = int(t[-1]/deltat), int(x[-1]/deltax)
    c = deltat/deltax**2

    match discretization:
        case 'Backward Euler':
            time = t[time_index] + deltat
        case 'Crank Nicholson':
            time = t[time_index] + deltat/2

    conditional_effects = np.array([0.0,0.0])
    for i, condition in enumerate(boundary_conditions):
        match condition:
            case 'Dirichlet':
                conditional_effects[i] = c*kappa(deltax,time)


print('yay')