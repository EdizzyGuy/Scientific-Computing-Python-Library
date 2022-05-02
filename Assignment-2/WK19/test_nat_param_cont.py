#%%
import sys
import os
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\\ld18821-emat30008'
sys.path.append(origin)

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numerical_continuation import find_limit_cycles
from utility import get_phase_portrait

'''
NATURAL PARAMETER CONTINUATION PRODUCES CORRECT GRAPHS FOR ALL 3
EXAMPLES GIVEN IN WORKSHEET 18
'''

# find when cubic = 0 for c varying from -2, 2
def cubic(x, c):
    return x**3 - x + c

# vary beta between 0 and 2
def hopf_bifurcation(t, U, beta):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 - u1 *(u1*u1 + u2*u2)
    u2_dot = u1 + beta*u2 - u2 *(u1*u1 + u2*u2)
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

# vary beta between -1 and 2 (start at 2)
def modified_hopf(t, U, beta):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 + u1 *(u1*u1 + u2*u2) - u1* (u1**2 + u2**2)**2
    u2_dot = u1 + beta*u2 + u2 *(u1*u1 + u2*u2) - u2* (u1**2 + u2**2)**2
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

# DOES NOT accepts multiple params to range
# added solver output option because most solvers will output ndarray
# could not find a way to specify methods name for root solver with class output
def natural_parameter_continuation(dXdt, init_guess, param_range=[0,1], root_solver=root, 
        precision=100, solver_object_output=False, args=()):

    params = np.linspace(param_range[0], param_range[1], precision)
    all_roots = np.empty((len(params), len(init_guess)))
    if root_solver is root or solver_object_output is True:
        for i, param in enumerate(params):
            sol = root_solver(dXdt, init_guess, args=(param, *args))
            if sol.success:
                all_roots[i, :] = sol.x
                init_guess = sol.x
            else:
                all_roots[i, :] = None
    else: # assumes output is ndarray

        for i, param in enumerate(params):
            sol = root_solver(dXdt, init_guess, solve_ivp, args=(param, *args))
            if isinstance(sol, np.ndarray):
                all_roots[i, :] = sol
                init_guess = sol
            else:
                all_roots[i,:] = None

    return all_roots, params
    
#%%
# TEST CUBIC ------------------------------------------------------------
init_guess = np.array([2])
param_range = [-2, 2]
# forward pass
sol_forw, params_forw = natural_parameter_continuation(cubic, init_guess, param_range)
plt.plot(params_forw, sol_forw, color='blue')
# mid pass
mid_guess = np.array([0])
sol_mid, params_mid = natural_parameter_continuation(cubic, mid_guess, param_range)
plt.plot(params_mid, sol_mid, color='blue')
#back pass
back_guess = np.array([-2])
param_range= [2,-2]
sol_back, params_back = natural_parameter_continuation(cubic, back_guess, param_range)
plt.plot(params_back, sol_back, color='blue')

plt.xlabel('parameter of cubic : c')
plt.ylabel('Real solution to cubic')
plt.title('Real solutions to $x^3 - x + c$\nw.r.t c')
plt.show()

#%%
# TEST HOPF ----------------------------------------------------------------
init_cond = np.array([1,1])
init_guess = np.append(init_cond, 10)
#forward pass
beta_forw = [0, 2]
sol_forw, params_forw = natural_parameter_continuation(hopf_bifurcation, init_guess, beta_forw, root_solver=find_limit_cycles)
rad_forw = np.linalg.norm(sol_forw[:,:-1], axis=1)
#backward pass
beta_back = [2, 0]
sol_back, params_back = natural_parameter_continuation(hopf_bifurcation, init_guess, beta_back, root_solver=find_limit_cycles)
rad_back = np.linalg.norm(sol_back[:,:-1], axis=1)

plt.plot(params_forw, rad_forw, color='red', label='Equilibrium')
plt.plot(params_back, rad_back, color='blue', label='Stable limit cycle')
plt.xlabel('Parameter : Beta')
plt.ylabel('Radius of orbit')
plt.title('Stable states of the norm hopf bifurcation w.r.t changing parameter')
plt.legend()
plt.show()

#%%
# TEST HOPF EXTENDED ---------------------------------------------------
init_cond = np.array([0,0])
init_guess = np.append(init_cond, 0)
beta_forw = [-1, 2]
sol_forw, params_forw = natural_parameter_continuation(modified_hopf, init_guess, beta_forw, root_solver=find_limit_cycles)
rad_forw = np.linalg.norm(sol_forw[:,:-1], axis=1)

init_cond = np.array([1,1])
init_guess = np.append(init_cond, 10)
beta_back = [2, -1]
sol_back, params_back = natural_parameter_continuation(modified_hopf, init_guess, beta_back, root_solver=find_limit_cycles)
rad_back = np.linalg.norm(sol_back[:,:-1], axis=1)

plt.plot(params_forw, rad_forw, color='red', label='forward pass')
plt.plot(params_back, rad_back, color='blue', label='backward pass')

plt.xlabel(r'Parameter : $\beta$')
plt.ylabel('Radius of orbit')
plt.title('Stable states of the mod. hopf bifurcation w.r.t changing parameter')
plt.legend()
plt.grid()
plt.show()