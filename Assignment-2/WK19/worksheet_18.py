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

# Q1
#   Write a code that performs natural parameter continuation, i.e., it simply increments 
# the a parameter by a set amount and attempts to find the solution for the new parameter value 
# using the last found solution as an initial guess.

# Q2
#   Write a code that performs pseudo-arclength continuation

#%%
#-----------------------------------------------------------------
# NATURAL PARAMETER CONTINUATION FOR CUBIC
# TODO make graphs on top of each other
fig = plt.figure(figsize=(10,6), dpi=100)
ax1 = plt.subplot(1,2,1)

C_forw = np.linspace(-2, 2, 100)
num_real_roots = []
for c in C_forw:
    roots = np.roots([c, -1, 0, 1])
    real_roots = 0
    for im in roots.imag:
        if im == 0:
            real_roots += 1
    num_real_roots.append(real_roots)
plt.plot(C_forw, num_real_roots)

plt.xlabel('parameter of cubic : c')
plt.ylabel('Number of real solution to cubic')
plt.title('Number of real solutions to $x^3 - x + c$\nw.r.t c')

ax2 = plt.subplot(1,2,2)
cubic_roots_forw = []
init_guess = 2
for c in C_forw:
    y = root(cubic, init_guess, args=(c,))
    if y.success == True:
        cubic_roots_forw.append(y.x)
        init_guess = y.x
    else:
        cubic_roots_forw.append(None)
plt.plot(C_forw, cubic_roots_forw, color='blue')

init_guess = 0
cubic_roots_mid = []
for c in C_forw:
    y = root(cubic, init_guess, args=(c,))
    if y.success == True:
        cubic_roots_mid.append(y.x)
        init_guess = y.x
    else:
        cubic_roots_mid.append(None)
plt.plot(C_forw, cubic_roots_mid, color='blue')

C_back = np.flip(C_forw)
init_guess = -2
cubic_roots_back = []
for c in C_back:
    y = root(cubic, init_guess, args=(c,))
    if y.success == True:
        cubic_roots_back.append(y.x)
        init_guess = y.x
    else:
        cubic_roots_back.append(None)
plt.plot(C_back, cubic_roots_back, color='blue')
 
plt.xlabel('parameter of cubic : c')
plt.ylabel('Real solution to cubic')
plt.title('Real solutions to $x^3 - x + c$\nw.r.t c')
plt.show()

#%%
#--------------------------------------------------------------------
# NATURAL PARAMETER CONTINUATION FOR HOPF BIFURCATION
# can plot radius of the limit cycle

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

plt.plot(BETA_forw, rad_forw, color='red', label='Equilibrium')
plt.plot(BETA_back, rad_back, color='blue', label='Stable limit cycle')

plt.xlabel('Parameter : Beta')
plt.ylabel('Radius of orbit')
plt.title('Stable states of the hopf bifurcation w.r.t changing parameter')
plt.legend()
plt.show()

# TESTED FOR ROOTSOLVERS THAT MIGHT BE BETTER THAN DEFAULT
#-BETTER 'lm'(didni get period)
#-krylov quick but wrong
# broyden1 takes long NO WORK
# broyden2 takes TOO long 
#-NO WORK 'df-sane' takes long


#%%
#--------------------------------------------------------------------
# NATURAL PARAMETER CONTINUATION FOR MOD HOPF BIFURCATION NORMAL FORM
# range beta from -1 to 2 (start : 2)
# can plot radius of the limit cycles

init_cond = np.array([1,1])
init_guess = np.append(init_cond, 10)

BETA_back = np.linspace(2, -1, 100)
rad_back = np.zeros(BETA_back.shape)
sol_back = []

BETA_forw = np.linspace(-1, 2, 100)
rad_forw = np.zeros(BETA_forw.shape)
sol_forw = []
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
plt.title('Stable states of the hopf bifurcation w.r.t changing parameter')
plt.legend()
plt.grid()
plt.show()
# %% 
# ---------------------------------------------------------------------------------------------
# Write a code that performs pseudo-arclength continuation

def secant(u2, u1):
    secant = u2 + (u2 - u1)
    return secant

# rough implementation of prediction correction approach
stable_states = []

beta1, beta2 = 2, 1.9

init_cond = np.array([1,1])
init_guess = np.append(init_cond, (10, beta1)) # append period guess and param value
u1 = find_limit_cycles(hopf_bifurcation, init_guess[:-1], solve_ivp, args=(init_guess[-1],))
stable_states.append(np.append(u1, beta1))

sec_guess = np.append(stable_states[0], beta2)
u2 = find_limit_cycles(hopf_bifurcation, sec_guess[:-1], solve_ivp, args=(sec_guess[-1],))
stable_states.append(np.append(u2, beta2))

fails = 0
while stable_states[-1][-1] > -1 and fails < 50: # while beta is bigger than -1
    secant_prediction = secant(stable_states[-1], stable_states[-2])
    solution = find_limit_cycles(hopf_bifurcation, secant_prediction[:-1], solve_ivp, args=(secant_prediction[-1],))
    if isinstance(solution, np.ndarray):
        state = np.append(solution, secant_prediction[-1])
        stable_states.append(state)
    else:
        fails += 1
    

stable_states = np.asarray(stable_states)
rads = np.linalg.norm(stable_states[:,:-2], axis=1)
betas = stable_states[:,-1]
plt.plot(betas, rads, color='blue')
plt.show()


#CUBIC EQUATION

