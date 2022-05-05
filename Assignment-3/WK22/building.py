import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

from solve_diffusionV1 import *
from utility import *
from math import pi

def test_forw_eul_matrix_kappaVariable_x():
    ''' Generates A_FE matrix for a VARIABLE diffusion coefficient, DEPENDANT ON X. Compares result to a 
    known (hand calculated) A_FE and asserts that they are the same.
    kappa function used is  :   kappa(x,t) = 2x
    '''
    T,L,mx,mt = 1,1,5,3

    def kappa(x):
        k =2*x
        return k

    t,x = get_grid_space(T,L,mt,mx)  

    # define true matrix
    A_true = np.zeros((1,mx-1,mx-1))
    A_true[:,0,:2] = np.array([1, 5])
    A_true[:,1,:3] = np.array([-5/3, -17/3,25/3])
    A_true[:,2,1:4] = np.array([5/3,-37/3,35/3])
    A_true[:,3,2:] = np.array([5,-19])

    # get matrix from predefined function
    A = forw_eul_pde_matrix_varKappa_x(t,x,kappa)

    return A

test_forw_eul_matrix_kappaVariable_x()
print('yay')