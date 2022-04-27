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

def secant_beta(u2, u1):
    secant_u = u2 + (u2 - u1)
    return secant_u

def secant(u2, u1, p2, p1):
    secant_u = u2 + (u2 - u1)
    secant_p = p2 + (p2 - p1)
    return secant_u, secant_p

#METHODS FOR NUMERCIAL CONTINUATION ONLY WORK IN 1D for param
# how to handle fails (will output None if find_limit_cycles fails)
# cannot form secant with None
def prediction_correction_parameter_continuation(dXdt, init_guess, param_range=[0,1], root_solver=root,
        init_delta_param=0.01, solver_object_output=False, args=()):
    
    # pass direction is unit positive if forward and unit negative if backward
    pass_direction = np.sign(param_range[-1] - param_range[0])
    #find first solution
    p1 = param_range[0]
    p2 = p1 + init_delta_param * pass_direction
    params = [p1, p2]
    if root_solver is root or solver_object_output is True:
        # 1st solutions
        sol = root_solver(dXdt, init_guess, args=(p1, *args))
        assert sol.success, 'initial guess has not found any solutions...\n Please input another initial guess'
        stable_solutions = [sol.x]
        #2nd solution
        sol = root_solver(dXdt, init_guess, args=(p2, *args))
        assert sol.success, 'second guess has not found any solutions...\n Please input another initial guess'
        stable_solutions.append(sol.x)

        #pred-corr approach:
        iter = 1
        while np.sign(param_range[-1] - params[-1]) is pass_direction:
            #generate secant
            secant_u, secant_p = secant(stable_solutions[-1], stable_solutions[-2], params[-1], params[-2])
            sol = root_solver(dXdt, secant_u, args=(secant_p, *args))
            if sol.success:
                stable_solutions.append(sol.x)
                params.append(secant_p)
                iter += 1
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {iter} of prediction correction numerical continuation')
                return stable_solutions, params
        
        return stable_solutions, params
    else: # assumes output is ndarray
        #1st solution
        u1 = root_solver(dXdt, init_guess, solve_ivp, args=(p1, *args))
        assert isinstance(u1, np.ndarray), 'initial guess has not found any solutions...\n Please input another initial guess'
        stable_solutions = [u1]
        #2nd solutions
        u2 = root_solver(dXdt, u1, solve_ivp, args=(p2, *args))
        assert isinstance(u2, np.ndarray), 'second guess has not found any solutions...\n Please input another initial guess'
        stable_solutions.append(u2)

        #with two solutions can implement pred-corr approach
        iter = 1
        while np.sign(param_range[-1] - params[-1]) is pass_direction:
            secant_u, secant_p = secant(stable_solutions[-1], stable_solutions[-2], params[-1], params[-2])
            u = root_solver(dXdt, secant_u, solve_ivp, args=(secant_p, *args))
            if isinstance(u, np.ndarray):
                stable_solutions.append(u)
                params.append(secant_p)
                iter += 1
            else:
                # ERROR MESSAGE HERE
                print(f'root solver has failed to find roots after iteration {iter} of prediction correction numerical continuation')
                return stable_solutions, params
        
        return stable_solutions, params

#%%
# rough implementation of prediction correction approach
stable_states = []

beta1, beta2 = 2, 1.9
init_delta_param=0.01

init_cond = np.array([1,1])
init_guess = np.append(init_cond, (10, beta1)) # append period guess and param value
u1 = find_limit_cycles(hopf_bifurcation, init_guess[:-1], solve_ivp, args=(init_guess[-1],))
stable_states.append(np.append(u1, beta1))

sec_guess = np.append(stable_states[0][:-1], beta2)
u2 = find_limit_cycles(hopf_bifurcation, sec_guess[:-1], solve_ivp, args=(sec_guess[-1],))
stable_states.append(np.append(u2, beta2))

fails = 0
while stable_states[-1][-1] > -1 and fails < 50: # while beta is bigger than -1
    secant_prediction = secant_beta(stable_states[-1], stable_states[-2])
    if secant_prediction[-1] == stable_states[-1][-1]:
        secant_prediction[-1] -= init_delta_param
    solution = find_limit_cycles(hopf_bifurcation, secant_prediction[:-1], solve_ivp, args=(secant_prediction[-1],))
    if isinstance(solution, np.ndarray):
        state = np.append(solution, secant_prediction[-1])
        stable_states.append(state)
    else:
        fails += 1
        t = 5 #must be initiated doesn't matter for equation
        deriv = hopf_bifurcation(t, secant_prediction[:-2], secant_prediction[-1])
        if np.all(np.isclose(deriv, 0, atol=1e-02)):
            # stable eq NOT LIMIT CYCLE
            eq_prediction = np.zeros(4)
            eq_prediction[-1] = secant_prediction[-1]
            solution = find_limit_cycles(hopf_bifurcation, eq_prediction[:-1], solve_ivp, args=(eq_prediction[-1],))
            deriv = hopf_bifurcation(t, solution[:-1], secant_prediction[-1])
            if not isinstance(solution, np.ndarray):
                print('complete failure')
                break
            elif np.all(np.isclose(deriv, 0)):
                # append state twice so that secant predictor will return equilibrium
                # DOESNT WORK CUS SECANT KEEPS PARAM CONSTANT
                state = np.append(solution, secant_prediction[-1])
                stable_states.append(state)
                stable_states.append(state)
    
        # will probably get stuck in a failing loop here
        # need to get find limit cycles to find equilibria
    

stable_states = np.asarray(stable_states)
rads = np.linalg.norm(stable_states[:,:-2], axis=1)
betas = stable_states[:,-1]
plt.plot(betas, rads, color='blue')
plt.title('Predictor corrector solving stable states of hopf bifurcation')
plt.show()



# %%
