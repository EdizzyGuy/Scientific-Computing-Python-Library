# %%
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


def secant(u2, u1, p2, p1):
    secant_u = u2 + (u2 - u1)
    secant_p = p2 + (p2 - p1)
    return secant_u, secant_p

def within_parameter_range(p, p_range, pass_direction):
    oracle = (np.sign(p_range[-1] - p) == pass_direction)
    return oracle

#METHODS FOR NUMERCIAL CONTINUATION ONLY WORK IN 1D
def prediction_correction_parameter_continuation(dXdt, init_guess, param_range=[0,1], root_solver=root,
        init_delta_param=0.01, solver_object_output=False, args=()):
    
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
            
            #TEST THIS:
            if not within_parameter_range(p, param_range, pass_direction):  # if true no solutions in param_range
                print('root solver has failed to find valid solutions within the parameter range\n',
                    'consider a different initial guess or root solver...')
                return params, solutions
            solutions.append(sol.x)
            params.append(p)

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
    else: # assumes output is ndarray
        #1st solution
        u1 = root_solver(dXdt, init_guess, solve_ivp, args=(p1, *args))
        assert isinstance(u1, np.ndarray), 'initial guess has not found any solutions...\n Please input another initial guess'
        solutions = [u1]
        #2nd solutions
        u2 = root_solver(dXdt, u1, solve_ivp, args=(p2, *args))
        assert isinstance(u2, np.ndarray), 'second guess has not found any solutions...\n Please input another initial guess'
        solutions.append(u2)

        #with two solutions can implement pred-corr approach
        iter = 1
        while np.sign(param_range[-1] - params[-1]) is pass_direction:
            secant_u, secant_p = secant(solutions[-1], solutions[-2], params[-1], params[-2])
            u = root_solver(dXdt, secant_u, solve_ivp, args=(secant_p, *args))
            if isinstance(u, np.ndarray):
                solutions.append(u)
                params.append(secant_p)
                iter += 1
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {iter} of prediction correction numerical continuation')
                return solutions, params
        
        return solutions, params

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
