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
    y = root(cubic, init_guess, args=(c))
    if y.success == True:
        cubic_roots_forw.append(y.x)
        init_guess = y.x
    else:
        cubic_roots_forw.append(None)
plt.plot(C_forw, cubic_roots_forw, color='blue')

init_guess = 0
cubic_roots_mid = []
for c in C_forw:
    y = root(cubic, init_guess, args=(c))
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
    y = root(cubic, init_guess, args=(c))
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

#--------------------------------------------------------------------
# NATURAL PARAMETER CONTINUATION FOR HOPF BIFURCATION
# can plot radius of the limit cycles

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
BETA_back = np.flip(BETA_forw)
#perform forward pass
init_guess = np.append(init_cond, 10)
for beta in BETA_forw:
    roots = find_limit_cycles(hopf_bifurcation, init_guess, solve_ivp, args=(2,))

# TESTED FOR ROOTSOLVERS THAT MIGHT BE BETTER THAN DEFAULT
#-BETTER 'lm'(didni get period)
#-krylov quick but wrong
# broyden1 takes long NO WORK
# broyden2 takes TOO long 
#-NO WORK 'df-sane' takes long

