import sys
import os
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\ld18821-emat30008'
sys.path.append(origin)

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numerical_continuation import find_limit_cycles
from utility import get_phase_portrait

# vary beta between 0 and 2
def hopf_bifurcation(t, U, beta):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 - u1 *(u1*u1 + u2*u2)
    u2_dot = u1 + beta*u2 - u2 *(u1*u1 + u2*u2)
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

# investigate properties of function
init_cond = np.array([1,1])
solve_for = np.linspace(0, 40, 400)
BETA = np.linspace(0, 2, 5)
for beta in BETA:
    print(beta)
    path = get_phase_portrait(hopf_bifurcation, init_cond, solve_for, solve_ivp, args=(beta,))
#always exhibits a stable limit cycle with increasing radius as beta increases
# lower beta values take longer to reach limit cycle
BETA_forw = np.linspace(0, 2, 100)
rad_forw = np.zeros(BETA_forw.shape)
sol_forw = []

BETA_back = np.flip(BETA_forw)
rad_back = np.zeros(BETA_back.shape)
sol_back = []
#perform forward pass
init_cond = np.array([1,1])
init_guess = np.append(init_cond, 10)
for i, beta in enumerate(BETA_forw):
    if i % 10 == 0:
        print(i)
    roots = find_limit_cycles(hopf_bifurcation, init_guess, solve_ivp, args=(beta,))
    if isinstance(roots, np.ndarray):
        init_guess = roots
        sol_forw.append(roots)
        rad_forw[i] = np.linalg.norm(init_guess[:-1])
    else:
        rad_forw[i] = None
        sol_forw.append(None)
plt.plot(BETA_forw, rad_forw, color='blue')

init_guess = np.append(init_cond, 10)
# back pass takes longer
for i, beta in enumerate(BETA_back):
    if i % 10 == 0:
        print(i)
    roots = find_limit_cycles(hopf_bifurcation, init_guess, solve_ivp, args=(beta,))
    if isinstance(roots, np.ndarray): 
        init_guess = roots
        sol_back.append(roots)
        rad_back[i] = np.linalg.norm(init_guess[:-1])
    else:
        rad_back[i] = None
        sol_back.append(None)
plt.plot(BETA_back, rad_back, color='blue')
plt.close()

# when doing forward pass solutions move to stable equilibria at origin
# also is finding solutions backwards in time!
# from inspecting phase portraits and adjusting time it is solved for it is clear
# that these forward passes are identifying unstable equilibria
# solutions found are indeed valid... on inspection can see each iterative solution collapsing
# into the unstable equilibria
# until 4th sol solutions are exhibiting limit cycles (the last valid solution did not find the
# correct period although this is probably because of tolerance (orbital trajectory is so small
# that almost anywhere on the solution would satisfy function G))  
i = 0
i += 1
roots = sol_forw[i]
solve_for = np.linspace(0, roots[-1], 100)
title = 'Showcasing potentially unstable equilibria at origin of hopf bifurcation\n' fr'$\beta$ : {BETA_forw[i]:.2f}'
path = get_phase_portrait(hopf_bifurcation, roots[:-1], solve_for, solve_ivp, title, args=(BETA_forw[i],))

plt.close()

#plot unstable in red and stable in blue
for i, sol in enumerate(sol_back):
    if i % 20 == 0:
        beta = BETA_back[i]
        rad = rad_back[i]
        solve_for = np.linspace(0, 100, 100)
        title = r'Limit cycle of Hopf bifurcation, $\beta$ = ' f'{beta:.2f}\nBackward Shooting says : radius {rad:.2E}'
        path = get_phase_portrait(hopf_bifurcation, sol[:-1], solve_for, solve_ivp, title, args=(beta,))

# forw sol quickly degenerate
# finds eq at origin through starting close to there
for i, sol in enumerate(sol_forw):
    if i % 20 == 0:
        beta = BETA_forw[i]
        rad = rad_forw[i]
        solve_for = np.linspace(0, 100, 100)
        title = r'Limit cycle of Hopf bifurcation, $\beta$ = ' f'{beta:.2f}\nForward Shooting says : radius {rad:.2E}'
        path = get_phase_portrait(hopf_bifurcation, sol[:-1], solve_for, solve_ivp,
            title, args=(beta,))

# DO THESE SOLUTIONS REACH UNSTABLE EQUILIBRIA DUE TO NUMERICAL ISSUES
#is it technically an unstable equilibiria
for i, sol in enumerate(sol_forw[:10]):
    beta = BETA_forw[i]
    rad = rad_forw[i]
    solve_for = np.linspace(0, 100, 100)
    title = f'Solution {i} of forward pass' r' $\beta$ = ' f'{beta:.2f}\nShooting says : radius {rad:.2E}'
    path = get_phase_portrait(hopf_bifurcation, sol[:-1], solve_for, solve_ivp,
            title, args=(beta,))

plt.plot(BETA_forw, rad_forw, color='red', label='Equilibrium')
plt.plot(BETA_back, rad_back, color='blue', label='Stable limit cycle')

plt.xlabel(r'Parameter : $\beta$')
plt.ylabel('Radius of orbit')
plt.title('Stable states of the hopf bifurcation w.r.t changing parameter')
plt.legend()
plt.show()
