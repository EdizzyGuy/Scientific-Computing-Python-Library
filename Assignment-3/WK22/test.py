from doctest import testfile
import sys
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)
from utility import *

import numpy as np
from solve_heat_eq3 import *
from solve_diffusionV1 import *
from scipy.integrate import solve_ivp
from math import pi

def test_forw_eul_matrix_kappaVariable_tx():
    ''' Generates A_FE matrix for a VARIABLE diffusion coefficient, DEPENDANT ON X AND T. Compares result to a 
        known (hand calculated) A_FE and asserts that they are the same.
        kappa function used is  :   kappa(x,t) = 2x - t 
        '''
    T,L,mx,mt = 1,1,5,3

    def kappa(t,x):
        k =2*x - t
        return k

    t,x = get_grid_space(T,L,mt,mx) 

    # define true matrix
    A_true = np.zeros((mt, mx+1,mx+1))
    A_true[0,1,:3] = np.array([-5/3, -17/3,25/3])
    A_true[0,1,1:4] = np.array([5/3,-37/3,35/3])
    A_true[0,2,2:5] = np.array([5,-19,15])
    A_true[0,3,3:] = np.array([5,-19,55/3])

    A_true[1,1,:3] = np.array([-40/9,-1/9,50/9])
    A_true[1,2,1:4] = np.array([-10/9,-61/9,80/9])
    A_true[1,3,2:5] = np.array([20/9,-121/9, 110/9])
    A_true[1,4,3:] = np.array([50/9,-181/9, 140/9])

    A_true[2,1,:3] = np.array([-65/9,49/9,25/9])
    A_true[2,2,1:4] = np.array([-35/9,-11/9,55/9])
    A_true[2,3,2:5] = np.array([-5/9,-71/9, 85/9])
    A_true[2,4,3:] = np.array([-5/9,-71/9, 115/9])

    # get matrix from predefined function
    A = forw_eul_pde_matrix_varKappa_tx(t,x,kappa)

    return np.all(np.isclose(A_true,A))

test_forw_eul_matrix_kappaVariable_tx()