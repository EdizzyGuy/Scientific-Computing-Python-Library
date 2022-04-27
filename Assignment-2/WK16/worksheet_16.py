#%%
import sys
import os
path = sys.path[0]
origin = os.path.dirname(os.path.dirname(path))
sys.path.append(origin)

import numpy as np
import time
import matplotlib.pyplot as plt
from ode import solve_ode
from utility import get_phase_portrait
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
#%%
'''
Q1
i   simulate predator prey equations

ii  what is behaviour in long time limit for
        b > 0.26, b < 0.26, b = 0.26

iii isolate a peridoic orbit
        what is its period and starting conditions

THESE ARE TESTS FOR NUMERICAL METHODS
    
Q2
    determine an appropiate phase condition
    
Q3
    Construct the shooting root-finding problem for the predator-prey example; 
    check that it can find the periodic orbit found in 1. 
    
Q4
    Generalise your code so that you can use arbitrary differential equations of arbitrary dimension 
    (assume they are always in first-order form).
    - How should a user pass the differential equations to your code?
    - How should a user pass the phase-condition to your code?
    - What options might a user want to have access to?
'''
#%%
# predator prey equations:
def dXdt(t, X, a, b, d):
    x = X[0]
    y = X[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b *y *(1 - y/x)

    dXdt = np.array([dxdt, dydt])
    return dXdt


a, b, d = 1, 0.26, 0.1
t = None
X = np.array([3, 7])
print(dXdt(t, X, a, b, d)) # function works
#%%
# simulate with b = 0.26
b = 0.26
initial_condition = np.random.uniform(size=2)
solve_for = np.linspace(0, 1000, 1000)
time_step = 0.01


# REPLACE THE BELOW WITH A TIMER FUNCTION
start_myode = time.time()
trajectory = solve_ode(dXdt, initial_condition, solve_for, deltat_max=time_step, method='RK4', args=(a, b, d))
end_myode = time.time()
my_ode_time = end_myode - start_myode

start_scipy = time.time()
traj = solve_ivp(fun=dXdt, t_span=[solve_for[0], solve_for[-1]], y0=initial_condition, t_eval=solve_for, max_step=time_step, args=(a,b,d))
end_scipy = time.time()
scipy_time = end_scipy - start_scipy

path = get_phase_portrait(dXdt, initial_condition, solve_for, title=f"Simulation of Predator-prey equations with arbitrary initial conditions : b={b}", 
    xlabel='prey', ylabel='predator', deltat_max=time_step, args=(a,b,d))
# CONVERGES TO STABLE LIMIT CYCLE FOR b = 0.26
#%%
# long time limit for b > 0.26
b = 0.3
get_phase_portrait(dXdt, initial_condition, solve_for, title=f"Simulation of Predator-prey equations with arbitrary initial conditions : b={b}", 
    xlabel='prey', ylabel='predator', deltat_max=time_step, args=(a,b,d))
# CONVERGES TO STABLE EQUILIBRIUM FOR b > 0.3
#%%
#long time limit for b < 0.26
b = 0.2
get_phase_portrait(dXdt, initial_condition, solve_for, title=f"Simulation of Predator-prey equations with arbitrary initial conditions : b={b}", 
    xlabel='prey', ylabel='predator', deltat_max=time_step, args=(a,b,d))
# STABLE LIMIT CYCLE -> it has been shown that solutions collapse to a stable equilibrium at b ~=0.27

#------------------------------------------------------------------------
#%%
# isolate a periodic orbit
# start at large time therfore will definitely be an orbit
solve_for = np.linspace(0, 1000)
trajectory = solve_ode(dXdt, initial_condition, solve_for, deltat_max=time_step, method='RK4', args=(a, b, d))
# BUG found bug where solve ode will always start at time = 0 even when specified to start at differet t
initial_orbit = trajectory[-1, :]

# through experimentation it is found that the period for these parameters is between 20 and 25
# by checking phase portrait for 100 units forward in time it is shown that oscillation between successive orbits
# is negligible
solve_for = np.linspace(0, 25, 10000)
orbital_trajectory = get_phase_portrait(dXdt, initial_orbit, solve_for, title=f"Simulation of Predator-prey equations with arbitrary initial conditions : b={b}", 
                        xlabel='prey', ylabel='predator', deltat_max=time_step, args=(a,b,d))
# numpys is close will elementwise evaluate vectors to see if they are close within a tolerance of 1e-08

close_positions = []
for index, position in enumerate(orbital_trajectory[1:, :]):
    if np.all(np.isclose(initial_orbit, position, atol=1e-04)):
        close_positions.append(index+1)
if len(close_positions) > 1:
    closest_index = None
    closest_mse = np.inf
    for index in close_positions:
        mse = np.square(orbital_trajectory[index] - initial_orbit).mean()
        if mse < closest_mse:
            closest_mse = mse
            closest_index = index
else:
    closest_index = close_positions[0]
# index 8325 found to be close to start position
# therefore can find period from solve_for array
period = solve_for[closest_index]
print(f'period found to be {period}')
# PERIOD FOUND TO BE 20.81
# STARTING CONDITION : initial_orbit, (a,b,d = 1, 0.2, 0.1)
#%%
#-----------------------------------------------------------------------
# determine an appropriate phase condition
#  x_dot = 0
def phase_condition(X, dXdt, args):
    X_dot = dXdt(t, X, *args)
    return X_dot[0]

#%%
#-----------------------------------------------------------------------
# Construct the shooting root-finding probleM for the predator-prey example; 
# check that it can find the periodic orbit found in 1.

def G(X, T, args):
   trajectory_T = solve_ode(dXdt, X, solve_for=[0, T], args=args)
   sol_T = trajectory_T[-1,:]
   G = X - sol_T
   return G

# functions G and phase condition verified

def root_finding_problem(X, G, phase_condition, dXdt, args):
    sol = np.zeros(shape=(3,))
    sol[:2] = G(X[:2], X[-1], args)
    sol[-1] = phase_condition(X[:2], dXdt, args)
    return sol
#%%
# testing to find period
test_init = np.zeros(shape=(3,))
test_init[:2] = initial_orbit
test_init[-1] = 15

roots = fsolve(root_finding_problem, test_init, args=(G, phase_condition, dXdt, (a, b, d)))
test_sol = roots[:2]
test_period = roots[-1]
solve_for = np.linspace(0, test_period, 100)
test_traj = get_phase_portrait(dXdt, test_sol, solve_for, title=f"Simulation of Predator-prey equations with arbitrary initial conditions : b={b}", 
                        xlabel='prey', ylabel='predator', deltat_max=time_step, args=(a,b,d))

if np.all(np.isclose(test_traj[0], test_traj[-1], atol=1e-04)):
    print('true orbital trajectory correctly found')

# by inspection of the two found orbits we can see they are the same
# also they have equal periods
#%%

#-------------------------------------------------------------------------
# Generalise your code so that you can use arbitrary differential equations of 
# arbitrary dimension (assume they are always in first-order form).
#   - How should a user pass the differential equations to your code?
#   - How should a user pass the phase-condition to your code?
#   - What options might a user want to have access to?

def find_limit_cycles(init_guess, dXdt, find_period=True, phase_condition=None, args=()):
    """ will find limit cycles - can input any requested phase_condition
    init guess must have state variables first then guess of period if period is known put known period in
    TESTS : test on higher dimensional question
            test on solution with known period"""
    t = None
    if find_period is False:
        phase_condition = lambda X, dXdt, args : 0
    elif (find_period is True) and (phase_condition is None):

        def phase_condition(X, dXdt, args):
            X_dot = dXdt(t, X, *args)
            return X_dot[0]

    def G(init_guess, args):
        X = init_guess[:-1]
        T = init_guess[-1]
        trajectory_T = solve_ode(dXdt, X, solve_for=[0, T], args=args)
        sol_T = trajectory_T[-1,:]
        G = X - sol_T
        return G

    def root_finding_problem(init_guess, G, phase_condition, dXdt, args):
        sol = np.zeros(shape=(3,))
        sol[:-1] = G(init_guess, args)
        sol[-1] = phase_condition(init_guess[:-1], dXdt, args)
        return sol    

    roots = fsolve(root_finding_problem, init_guess, args=(G, phase_condition, dXdt, args))
    return roots

init_guess = np.random.uniform(size=3)
init_guess[-1] *= 40
roots = find_limit_cycles(test_init, dXdt, args=(a,b,d))
print(f'period found to be {roots[-1]}')
# correctly finds roots on example
#%%
# test to see if code works when period is known
period = roots[-1]
init_guess = np.random.uniform(size=3)
init_guess[-1] = period
roots = find_limit_cycles(init_guess, dXdt, find_period=False, args=(a,b,d))
path = get_phase_portrait(dXdt, roots[:-1], solve_for=np.linspace(0, period, 100), args=(a,b,d))
# as illustrated by phase portrait function can also handle known periods
print(f'period found to be {roots[-1]}')
# %%
