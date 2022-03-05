import Euler
import numpy as np
import matplotlib.pyplot as plt

''' TEST IDEAS
solve ode to a certain point
solve ode with multiple points in between AND PRINT ERRORS'''

'''Q1 : Solve ODE x_dot = x, with initial condition x(0) = 1.
solve with different time steps'''


def dx_dt(x):
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

i = 0
for time_step in time_steps:
    result_euler = Euler.solve_ode(dx_dt, initial_condition, time, deltat_max=time_step, method='Euler')
    result_RK4 = Euler.solve_ode(dx_dt, initial_condition, time, deltat_max=time_step, method='RK4')

    errors_euler[i] = result_euler[1][0] - true_ans
    errors_RK4[i] = result_RK4[1][0] - true_ans
    if i % 10 == 0:
        print(i)
    i += 1

fig = plt.figure(figsize=(8, 6), dpi=100)
fig.suptitle('Effectiveness of solve_ode function with different iterative methods')

ax1 = plt.subplot(1, 2, 1)
plt.loglog(time_steps, np.abs(errors_euler), color='blue', linewidth=2.5, linestyle='-')
ax1.set_xlabel(r'Maximum step size')
ax1.set_ylabel(r'Error in approximation to x(1)')
ax1.set_title(r"Euler's method")
plt.grid()

ax2 = plt.subplot(1, 2, 2)
plt.loglog(time_steps, np.abs(errors_RK4), color='blue', linewidth=2.5, linestyle='-')
ax2.set_xlabel(r'Maximum step size')
# maybe doesnt need y axis label
ax2.set_title(r"classic Runge-Kutta method")
plt.grid()

plt.show()
while True:
    print('eat shit')


'''
error = result - true_ans
print('Individual errors are : ', error)
print('Average error is :', np.average(error))
# HAS REDUCING ACCURACY AS ESTIMATES DIVERGE FROM INITIAL POINT
'''
'''Produce a (nicely formatted) plot with double logarithmic scale showing how 
the error depends on the size of the timestep Δt'''
# DOUBLE LOGARITHMIC SCALE IS LOG LOG GRAPH

'''
WORKS WITH 2 TIMES (time = [0, 1])

seems to screw up after first guess'''
