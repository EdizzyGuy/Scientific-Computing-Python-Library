import sys
import os

path = sys.path[0]
parent = os.path.dirname(path)
sys.path.append(parent)

import ode
import numpy as np
import matplotlib.pyplot as plt

''' TEST IDEAS
solve ode to a certain point
solve ode with multiple points in between AND PRINT ERRORS'''

'''
Q1 : Solve ODE x_dot = x, with initial condition x(0) = 1.
solve with different time steps

Q2 : Produce a (nicely formatted) plot with double logarithmic 
scale showing how the error depends on the size of the timestep Δt\Delta tΔt.
do for both Euler step and RK4'''


def dx_dt(t, x):
    x_dot = x
    return x_dot

time_steps = np.logspace(-4, 0, 42)
# start at 10^-16 as this is the smallest number matlab can represent IS IT
# -16 is too much will req 10^16 steps to complete a unit
# end at 10^0 = 1 since largest timestep from 0 to 1 is 1.

time = np.array([0, 1])
initial_condition = np.array(1)
true_ans = np.exp(time[1])
errors_euler = np.zeros(shape=time_steps.shape)
errors_RK4 = np.zeros(shape=time_steps.shape)
errors_RK45 = np.zeros(shape=time_steps.shape)

i = 0
for time_step in time_steps:
    result_euler = ode.solve_ode(dx_dt, initial_condition, time, deltat_max=time_step, method=ode.euler_step)
    result_RK4 = ode.solve_ode(dx_dt, initial_condition, time, deltat_max=time_step, method=ode.RK4_step)
    result_RK45 = ode.solve_ode(dx_dt, initial_condition, time, deltat_max=time_step, method=ode.RK45_step)

    errors_euler[i] = result_euler[1][0] - true_ans
    errors_RK4[i] = result_RK4[1][0] - true_ans
    errors_RK45[i] = result_RK45[1][0] - true_ans
    i += 1

fig = plt.figure(figsize=(8, 6), dpi=100)
fig.suptitle('Effectiveness of solve_ode function with different iterative methods', fontweight='bold')

ax1 = plt.subplot(1, 3, 1)
plt.loglog(time_steps, np.abs(errors_euler), color='blue', linewidth=2.5, linestyle='-')
ax1.set_xlabel(r'Maximum step size')
ax1.set_ylabel(r'Error in approximation to x(1)')
ax1.set_title(r"Euler's method")
plt.grid()

ax2 = plt.subplot(1, 3, 2)
plt.loglog(time_steps, np.abs(errors_RK4), color='blue', linewidth=2.5, linestyle='-')
ax2.set_xlabel(r'Maximum step size')
# maybe doesnt need y axis label
ax2.set_title("classic Runge-Kutta\nmethod")
plt.grid()

ax3 = plt.subplot(1, 3, 3)
plt.loglog(time_steps, np.abs(errors_RK45), color='blue', linewidth=2.5, linestyle='-')
ax3.set_xlabel(r'Maximum step size')
# maybe doesnt need y axis label
ax3.set_title("Runge-Kutta-Fehlberg\nmethod")
plt.grid()

min = np.min([ax3.get_ylim()[0],ax1.get_ylim()[0],ax2.get_ylim()[0]])
max = np.max([ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1]])
y_lim = (min, max)
ax1.set_ylim(y_lim)
ax2.set_ylim(y_lim)
ax3.set_ylim(y_lim)

plt.show()
#fig.savefig('errors_Euler_rk4_rk45.png')

''' How does the error depend on Δt\Delta tΔt now? How does this compare with the error for the Euler method (put this in the
same plot)? '''


'''
error = result - true_ans
print('Individual errors are : ', error)
print('Average error is :', np.average(error))
# HAS REDUCING ACCURACY AS ESTIMATES DIVERGE FROM INITIAL POINT
'''

'''
WORKS WITH 2 TIMES (time = [0, 1])

seems to screw up after first guess'''
