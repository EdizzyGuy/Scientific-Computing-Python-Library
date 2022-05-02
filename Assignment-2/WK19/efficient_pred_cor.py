# %%
import sys
import os
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\\ld18821-emat30008'
sys.path.append(origin)

import time
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numerical_continuation import find_limit_cycles
from utility import get_phase_portrait

def secant(u2, u1):
    secant_u = u2 + (u2 - u1)
    return secant_u

def within_parameter_range(p, p_range, pass_direction):
    oracle = (np.sign(p_range[-1] - p) == pass_direction)
    return oracle

def get_consequative_solutions(dXdt, init_guess, params, root_solver, args):
# specifically for root_solver=root
# returns index of last position
    all_roots = np.full((len(params), len(init_guess)), np.nan) # initialise solution array
    consequative_solutions = 0
    if root_solver is root:
        for i, p in enumerate(params):
            sol = root_solver(dXdt, init_guess, args=(p, *args))
            if sol.success:
                all_roots[i,:] = sol.x
                init_guess = sol.x
                consequative_solutions += 1
                if consequative_solutions == 2:
                    break
            else:
                all_roots[i,:] = None
                consequative_solutions = 0
    else:
        for i,p in enumerate(params):
            sol = root_solver(dXdt, init_guess, args=(p, *args))
            if isinstance(sol, np.ndarray):
                all_roots[i,:] = sol
                init_guess = sol
                consequative_solutions += 1
                if consequative_solutions == 2:
                    break
            else:
                all_roots[i,:] = None
                consequative_solutions = 0

    return all_roots, i+1

#results = continuation(myode,  # the ODE to use
 #   x0,  # the initial state
 #   par0,  # the initial parameters
 #   vary_par=0,  # the parameter to vary
 #   step_size=0.1,  # the size of the steps to take
 #   max_steps=100,  # the number of steps to take
 #   discretisation=shooting,  # the discretisation to use
 #   solver=scipy.optimize.fsolve)  # the solver to use
def prediction_correction_parameter_continuation(dXdt, init_guess, param_range=[0,1], root_solver=root,
        precision=100, solver_object_output=False, args=()):
    ''' '''
#make sure arguements are given in correct type
    params = np.linspace(param_range[0], param_range[1], precision)
    all_roots, last_pos = get_consequative_solutions(dXdt, init_guess, params, root_solver, args)
    if last_pos == len(params):
        print('root solver has failed to find valid solutions within the parameter range\n',
                    'consider a different initial guess or root solver...')
        return all_roots, params

    #pred-corr approach:
    if root_solver is root:
        for i, secant_p in enumerate(params[last_pos:]):
            index = i + last_pos  # index of first position in array that has not been filled with solution
            #generate secant
            secant_u = secant(all_roots[index-1], all_roots[index-2])
            sol = root_solver(dXdt, secant_u, args=(secant_p, *args))
            if sol.success:
                all_roots[index] = sol.x
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {i} of prediction correction numerical continuation')
                print(f'Parameter varied from {params[last_pos-2]} to {params[index-1]}')
                break
    else:
        for i, secant_p in enumerate(params[last_pos:]):
            index = i + last_pos  # index of first position in array that has not been filled with solution
            #generate secant
            secant_u = secant(all_roots[index-1], all_roots[index-2])
            sol = root_solver(dXdt, secant_u, args=(secant_p, *args))
            if isinstance(sol, np.ndarray):
                all_roots[index] = sol
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {i} of prediction correction numerical continuation')
                print(f'Parameter varied from {params[last_pos-2]} to {params[index-1]}')
                break
    
    return all_roots[:index], params[:index]

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

#%%
# TEST CUBIC ------------------------------------------------------------
init_guess = np.array([2])
param_range = [-2, 2]
# forward pass
sol_forw, params_forw = prediction_correction_parameter_continuation(cubic, init_guess, param_range)
plt.plot(params_forw, sol_forw, color='blue')
# mid pass
mid_guess = np.array([0])
sol_mid, params_mid = prediction_correction_parameter_continuation(cubic, mid_guess, param_range)
plt.plot(params_mid, sol_mid, color='blue')
#back pass
back_guess = np.array([-2])
param_range= [2,-2]
sol_back, params_back = prediction_correction_parameter_continuation(cubic, back_guess, param_range)
plt.plot(params_back, sol_back, color='blue')

plt.xlabel('parameter of cubic : c')
plt.ylabel('Real solution to cubic')
plt.title('Real solutions to $x^3 - x + c$\nw.r.t c')
plt.show()

# %%
# TEST HOPF ----------------------------------------------------------------
init_cond = np.array([1,1])
init_guess = np.append(init_cond, 10)
#forward pass
beta_forw = [0, 2]
sol_forw, params_forw = prediction_correction_parameter_continuation(hopf_bifurcation, init_guess, beta_forw, root_solver=find_limit_cycles)
rad_forw = np.linalg.norm(sol_forw[:,:-1], axis=1)
#backward pass
beta_back = [2, 0]
sol_back, params_back = prediction_correction_parameter_continuation(hopf_bifurcation, init_guess, beta_back, root_solver=find_limit_cycles)
rad_back = np.linalg.norm(sol_back[:,:-1], axis=1)

plt.plot(params_forw, rad_forw, color='red', label='Equilibrium')
plt.plot(params_back, rad_back, color='blue', label='Stable limit cycle')
plt.xlabel('Parameter : Beta')
plt.ylabel('Radius of orbit')
plt.title('Stable states of the norm hopf bifurcation w.r.t changing parameter')
plt.legend()
plt.show()


# %%
