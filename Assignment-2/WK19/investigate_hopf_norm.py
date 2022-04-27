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

# vary beta between -1 and 2 (start at 2)
# PERIOD IS STILL 2PI
def modified_hopf(t, U, beta):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 + u1 *(u1*u1 + u2*u2) - u1* (u1**2 + u2**2)**2
    u2_dot = u1 + beta*u2 + u2 *(u1*u1 + u2*u2) - u2* (u1**2 + u2**2)**2
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot


# investigate properties of function
# forward pass
init_cond = np.array([1,1])
solve_for = np.linspace(0, 15, 100)
BETA = np.linspace(2, -1, 7)
for beta in BETA:
    title = 'Solution to modified Hopf bifurcation\nBACKWARD PASS\n' r'$\beta$ = ' f'{beta} | initial condition = {init_cond}'
    path = get_phase_portrait(modified_hopf, init_cond, solve_for, solve_ivp, title, args=(beta,))

# stable limit cycle for beta going from 2 to 0, then solutions collapse into stable equilibria in the
# form of a spiral

#backward pass
BETA = np.flip(BETA)
for beta in BETA:
    title = 'Solution to modified Hopf bifurcation\nFORWARD PASS\n' r'$\beta$ = ' f'{beta} | initial condition = {init_cond}'
    path = get_phase_portrait(modified_hopf, init_cond, solve_for, solve_ivp, title, args=(beta,))
# no reason to start at beta = 2


init_cond = np.array([0.1, 0.1])
init_guess = np.append(init_cond, 10)
beta = -1
# TEST ROOT ARGS WORKING
root_args = [{'maxfev' : 0}, {'maxiter' : 400}]
roots = find_limit_cycles(modified_hopf, init_guess, solve_ivp, root_args=root_args, args=(beta,))
# takes a long time to find solution because solution probably takes long time to reach eq at 0 
#probably why we were told to do a backward pass
if roots[-1] > 100:
    roots[-1] = 100
title = r'stable equilibria at origin for $\beta$ = ' f'{beta}'
path = get_phase_portrait(modified_hopf, roots[:-1], np.linspace(0, roots[-1], 100), title=title, args=(beta,)) # solutions always spirallin
# once beta is -ve solutions will spiral to origin but trajectories starting on beta=0 orbit
# spiral more slowly giving the illusion of finding a limit cycle
# stops calculating roots well for beta below ~-0.5
path = get_phase_portrait(modified_hopf, roots[:-1], np.logspace(0, 1, 100), title='a better plot', args=(beta,)) #a better plot
#test behaviour near origin
origin =np.array([1e-06, 1e-06])
#solve_ode works backwards in time
path = get_phase_portrait(modified_hopf, origin, np.linspace(0, 10, 100), title='Trajectory of solution starting ~ at origin', args=(2,))
# speed that trajectory reaches limit cycle inversely proportional to beta
path = get_phase_portrait(modified_hopf, origin*2e06, np.linspace(1, 100, 100), title='Trajectory of solution starting ~ at origin', args=(0,))
#encounters run time errors when starting pos is > [4.9,4.9]
#also changes from moving towards limit cycle at r = 1 to limit cycle r=(wherever it starts)

# perform natural parameter continuation
BETA_forw = np.linspace(-1, 2, 100)
rad_forw = np.zeros(BETA_forw.shape)
sol_forw = []

BETA_back = np.flip(BETA_forw)
rad_back = np.zeros(BETA_back.shape)
sol_back = []

init_cond = np.array([1,1])
init_guess = np.append(init_cond, 10)
# this is computationally more difficult
# limit cycles not found 4 times
fails = 0
for i, beta in enumerate(BETA_back):
    if i % 10 == 0:
        print(i)
    roots = find_limit_cycles(modified_hopf, init_guess, solve_ivp, args=(beta,))
    if isinstance(roots, np.ndarray): 
        init_guess = roots
        sol_back.append(roots)
        rad_back[i] = np.linalg.norm(init_guess[:-1])
    else:
        fails += 1
        rad_back[i] = None
        sol_back.append(None)
# fails to find solution 4 times
plt.plot(BETA_back, rad_back, color='blue', label='Stable limit cycle')
plt.show()

test = sol_back[1]
init_test = test[:-1]
period_test = test[-1]
path = get_phase_portrait(modified_hopf, init_test, np.linspace(0, period_test /2, 100), solve_ivp, 
    args=(BETA_back[1],))
    # PERIOD IS 2PI

#perform forward pass (cheating by giving know sol)
init_cond = np.array([0,0])
init_guess = np.append(init_cond, 0)
for i, beta in enumerate(BETA_forw):
    if i % 10 == 0:
        print(i)
    roots = find_limit_cycles(modified_hopf, init_guess, solve_ivp, args=(beta,))
    if isinstance(roots, np.ndarray):
        init_guess = roots
        sol_forw.append(roots)
        rad_forw[i] = np.linalg.norm(init_guess[:-1])
    else:
        rad_forw[i] = None
        sol_forw.append(None)
# just finds 0's


plt.plot(BETA_forw, rad_forw, color='red', label='forward pass')
plt.plot(BETA_back, rad_back, color='blue', label='backward pass')

plt.xlabel(r'Parameter : $\beta$')
plt.ylabel('Radius of orbit')
plt.title('Stable states of the hopf bifurcation w.r.t changing parameter\n NATURAL PARAMETER CONTINUATION')
plt.legend()
plt.grid()
plt.show()

