import sys
import os
path = sys.path[0]
origin = os.path.dirname(os.path.dirname(path))
print(origin)
sys.path.append(origin)

import numpy as np
from utility import get_phase_portrait
from numerical_continuation import find_limit_cycles
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import ode
import matplotlib.pyplot as plt

"""
(1) : hopf bifurcation equation
(2) : u1 = sqrt(beta)*cos(t+theta)
      u2 = sqrt(beta)*sin(t+theta) for sigma = 1 and theta = phase
Q1
Adapt your shooting code from last week to work with general ODEs (such as (1)). Define and document
an appropriate API for interacting with your shooting code.

Q2
Write a test script that runs your shooting code on (1) and checks it against the explicit solution (2).

Q3
As part of the same test script, test your shooting code against other examples where analytical solutions are known.

Q4
Add tests to check that your code handles errors gracefully (NICE ERROR MSGS). Consider errors such as
    - providing initial values where the dimensions don't match those of the ODE, or
    - providing inputs such that the numerical root finder does not converge.
"""

def hopf_bifurcation(t, U, beta, sigma):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 + sigma *u1 *(u1*u1 + u2*u2)
    u2_dot = u1 + beta*u2 + sigma *u2 *(u1*u1 + u2*u2)
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

beta, sigma = 2,-1

init_cond = np.array([2,-2])

sci = solve_ivp(hopf_bifurcation, [0,100], init_cond, args=(beta, sigma))
#scipy better for solving this kind of problem
u = sci.y
plt.plot(u[0], u[1])
plt.show()

som = get_phase_portrait(hopf_bifurcation, init_cond, solve_for=np.linspace(0,200,1000), deltat_max=1e-2, args=(beta, sigma))
# many problems with this equation have found out that it grows exponentially under many conditions
# TODO : error message for overflow and graceful exit
# handles negative and 0 sigma well
# seems that solutions are acctually decaying to 0 and pc lacks precision for such small values
# was just using sigma = 1 instead of -1 :/
# solutions for sigma < 0 seem to be in a growing (unstable) limit cycle 
# -> probably due to variation in numerical solutions
# as sigma approaches 0 radius of periodic limit cycle grows exponentially as the solution is in the form of a spiral locating it
# at sigma = 0 the circle is infinitely large and so the solution is simply a spiral to infinity
# for larger sigma straight exponential is growth bifurcation is here

init_guess = np.zeros(3)
init_guess[:-1] = init_cond
init_guess[-1] = 5
roots = find_limit_cycles(init_guess, hopf_bifurcation, args=(beta, sigma))
# QUITE SLOW in some cases a minute
# check to verify this is a correct limit cycle
path = get_phase_portrait(hopf_bifurcation, roots[:-1], solve_for=np.linspace(0,roots[-1], 100), args=(beta, sigma))
if np.all(np.isclose(path[0], path[-1])):
    print('start and end points of orbit are the same')
path = get_phase_portrait(hopf_bifurcation, roots[:-1], solve_for=np.linspace(0,roots[-1]*0.9, 100), args=(beta, sigma))   
# as shown the correct period is found (and not a scalar multiple)

#---------------------------------------------------------------------------------------------------------------------------------------
# Write a test script that runs your shooting code on (1) and checks it against the explicit solution (2).

# from analytical solution we know that period should be 2*pi
# will produce a circle therefore can check radius and centre, will be centred around origin

def test_hopf(hopf_bifurcation):
    beta, sigma = 2,-1
    args = (beta, sigma)
    # from analytical solution
    anal_period = 2* np.pi
    anal_radius = np.sqrt(beta)

    init_cond = np.random.normal(size=2) * 9
    period_guess = np.random.uniform(low=5, high=25, size=1)
    #changed period guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
    init_guess = np.concatenate((init_cond, period_guess))

    roots = find_limit_cycles(init_guess, hopf_bifurcation, args=(beta, sigma))
    num_period = roots[-1]
    num_radius = np.linalg.norm(roots[:-1])

    # check if period is a multiple of the analytical sol
    period_multiplicity = num_period / anal_period
    period_multiple = np.isclose(period_multiplicity % 1, 1) or np.isclose(period_multiplicity % 1, 0)
    print(f'Numerical and analytical values for period of orbit are integer multiples : {period_multiple}')
    print(f'Numerical and analytical values for radius of orbit are equal : {np.isclose(num_radius, anal_radius)}')
    return init_guess, roots

# code has shown to find limit cycles with period a multiple of the real period
test = test_hopf(hopf_bifurcation)
path = get_phase_portrait(hopf_bifurcation, test[0][:-1], solve_for=np.linspace(0, test[1][-1], 100), args=(2, -1))
# works ~80% of the time in testing

#-------------------------------------------------------------------------------------
# As part of the same test script, test your shooting code against other examples where 
# analytical solutions are known. Vary the number of dimensions. For example, try

def hopf_extended(t, U, beta=2, sigma=-1):
    t = None

    U_dot = np.zeros(3)
    U_dot[:-1] = hopf_bifurcation(t, U[:-1], beta, sigma)
    U_dot[-1] = - U[-1]

    return U_dot

# should find the same solutions as u3 will decay to 0 as t -> oo

def test_ext_hopf():
    beta, sigma = 2,-1
    args = (beta, sigma)
    # from analytical solution
    anal_period = 2* np.pi
    anal_radius = np.sqrt(beta)

    init_cond = np.array([1,1,1])
    #np.random.normal(size=3) * 9
    period_guess = np.random.uniform(low=5, high=25, size=1)
    #changed period guess from normal distribution to uniform as sometimes period guess would converge to sol of T=0
    init_guess = np.concatenate((init_cond, period_guess))

    roots = find_limit_cycles(init_guess, hopf_extended, args=(beta, sigma))
    numer_period = roots[-1]
    numer_radius = np.linalg.norm(roots[:-1])

    # check if period is a multiple of the analytical sol
    period_multiplicity = numer_period / anal_period
    period_multiple = np.isclose(period_multiplicity % 1, 1) or np.isclose(period_multiplicity % 1, 0)
    print(f'Numerical and analytical values for period of orbit are integer multiples : {period_multiple}')
    print(f'Numerical and analytical values for radius of orbit are equal : {np.isclose(numer_radius, anal_radius)}')
    return init_guess, roots

roots = test_ext_hopf()
# works sometimes but solutions diverge for long time and that may be causing issues for fsolve
# maybe try smaller step sizes

#-----------------------------------------------------------------------------------------------------
# TODO MOAR TESTS
# Add tests to check that your code handles errors gracefully. Consider errors such as
  #  - providing initial values where the dimensions don't match those of the ODE, or
  #  - providing inputs such that the numerical root finder does not converge.

