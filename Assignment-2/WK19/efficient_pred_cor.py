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


def secant(u2, u1):
    secant_u = u2 + (u2 - u1)
    return secant_u

def within_parameter_range(p, p_range, pass_direction):
    oracle = (np.sign(p_range[-1] - p) == pass_direction)
    return oracle

if root_solver is root or solver_object_output is True:
        # 1st solutions
        consequative_solutions = 0
        # need 2 consequative solutions for secant
        while consequative_solutions < 2: 
            sol = root_solver(dXdt, init_guess, args=(p, *args))
            # increment parameter until valid solution found
            while not sol.success and within_parameter_range(p, param_range, pass_direction):
                consequative_solutions = 0
                params.append(p)
                solutions.append(None)

                p += init_delta_param * pass_direction
                sol = root_solver(dXdt, init_guess, args=(p, *args))
            
            solutions.append(sol.x)
            params.append(p)
            #TEST THIS:
            if not within_parameter_range(p, param_range, pass_direction):  # if true no solutions in param_range
                print('root solver has failed to find valid solutions within the parameter range\n',
                    'consider a different initial guess or root solver...')
                return params, solutions

            consequative_solutions += 1
            p += init_delta_param * pass_direction
            init_guess = solutions[-1]
def get_consequative_solutions(dXdt, init_guess, params, root_solver, args):
# specifically for root_solver=root
# returns index of last position
    all_roots = np.empty((len(params), len(init_guess))) # initialise solution array
    consequative_solutions = 0
    for i, p in enumerate(params):
        sol = root_solver(dXdt, init_guess, args=(p, *args))
        if sol.success:
            all_roots[i,:] = sol.x
            init_guess = sol.x
            consequative_solutions += 1
        elif consequative_solutions == 2:
            print(f'i is {i}')
            break
        else:
            all_roots[i,:] = None
            consequative_solutions = 0
    
    print(f'i is {i}')
    return all_roots, i+1

#METHODS FOR NUMERCIAL CONTINUATION ONLY WORK IN 1D
#algorithm seems to be really inefficient for the hopf bifurcation
# takes 65 sec for a forw pass and 80s for backward
def prediction_correction_parameter_continuation(dXdt, init_guess, param_range=[0,1], root_solver=root,
        precision=100, solver_object_output=False, args=()):

    params = np.linspace(param_range[0], param_range[1], precision)
    all_roots, last_pos = get_consequative_solutions(dXdt, init_guess, params, root_solver, args)
    if index == len(params):
        print('root solver has failed to find valid solutions within the parameter range\n',
                    'consider a different initial guess or root solver...')
        return all_roots, params

    #pred-corr approach:
    iter = 1
    for i, secant_p in enumerate(params[last_pos+1]):
        #generate secant
        secant_u = secant(all_roots[i-1], all_roots[i-2])
        sol = root_solver(dXdt, secant_u, args=(secant_p, *args))
        if sol.success:
            all_roots[i] = sol.x
            iter += 1
        else:
            # ERROR MESSAGE HERE
            print(f'root solver has failed to find roots after iteration {iter} of prediction correction numerical continuation')
            return solutions, params
    
    return solutions, params
    
    
    # pass direction is unit positive if forward and unit negative if backward
    pass_direction = np.sign(param_range[-1] - param_range[0])
    #find first solution
    p = param_range[0]

    params = []
    solutions = []
    if root_solver is root or solver_object_output is True:
        # 1st solutions
        consequative_solutions = 0
        # need 2 consequative solutions for secant
        while consequative_solutions < 2: 
            sol = root_solver(dXdt, init_guess, args=(p, *args))
            # increment parameter until valid solution found
            while not sol.success and within_parameter_range(p, param_range, pass_direction):
                consequative_solutions = 0
                params.append(p)
                solutions.append(None)

                p += init_delta_param * pass_direction
                sol = root_solver(dXdt, init_guess, args=(p, *args))
            
            solutions.append(sol.x)
            params.append(p)
            #TEST THIS:
            if not within_parameter_range(p, param_range, pass_direction):  # if true no solutions in param_range
                print('root solver has failed to find valid solutions within the parameter range\n',
                    'consider a different initial guess or root solver...')
                return params, solutions

            consequative_solutions += 1
            p += init_delta_param * pass_direction
            init_guess = solutions[-1]
        # once loop is broken two consequative solutions are guaranteed and so we can proceed
        assert (solutions[-1] is not None) and (solutions[-2] is not None)

        #pred-corr approach:
        iter = 1
        while within_parameter_range(p, param_range, pass_direction):
            #generate secant
            secant_u, secant_p = secant(solutions[-1], solutions[-2], params[-1], params[-2])
            sol = root_solver(dXdt, secant_u, args=(secant_p, *args))
            if sol.success:
                solutions.append(sol.x)
                params.append(secant_p)
                iter += 1
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {iter} of prediction correction numerical continuation')
                return solutions, params
        
        return solutions, params
    else: # assumes output of root solver is ndarray
        #1st solution
        # 1st solutions
        consequative_solutions = 0
        # need 2 consequative solutions for secant
        while consequative_solutions < 2: 
            sol = root_solver(dXdt, init_guess, solve_ivp, args=(p, *args))
            # increment parameter until valid solution found
            while (not isinstance(sol, np.ndarray)) and within_parameter_range(p, param_range, pass_direction):
                consequative_solutions = 0
                params.append(p)
                solutions.append(sol)

                p += init_delta_param * pass_direction
                sol = root_solver(dXdt, init_guess, solve_ivp, args=(p, *args))
            
            solutions.append(sol)
            params.append(p)
            #TEST THIS:
            if not within_parameter_range(p, param_range, pass_direction):  # if true no solutions in param_range
                print('root solver has failed to find valid solutions within the parameter range\n',
                    'consider a different initial guess or root solver...')
                break
            
            consequative_solutions += 1
            p += init_delta_param * pass_direction
            init_guess = solutions[-1]
        # once loop is broken two consequative solutions are guaranteed and so we can proceed
        assert (solutions[-1] is not None) and (solutions[-2] is not None)

        #pred-corr approach:
        iter = 1
        #generate secant
        secant_u, secant_p = secant(solutions[-1], solutions[-2], params[-1], params[-2])
        while within_parameter_range(secant_p, param_range, pass_direction):           
            sol = root_solver(dXdt, secant_u, args=(secant_p, *args))
            if isinstance(sol, np.ndarray):
                solutions.append(sol)
                params.append(secant_p)
                iter += 1
                secant_u, secant_p = secant(solutions[-1], solutions[-2], params[-1], params[-2])
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {iter} of prediction correction numerical continuation')
                break
        
    solutions = np.asarray(solutions)
    params = np.asarray(params)
    return solutions, params