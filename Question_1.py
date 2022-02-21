import Euler
import numpy as np

''' TEST IDEAS
solve ode to a certain point
solve ode with multiple points in between'''

def dx_dt(x):
    x_dot = x
    return x_dot


print('we have started')
time = np.linspace(0, 10)
#print(time)
initial_condition = np.array(1)
result = Euler.solve_ode(dx_dt, initial_condition, time, deltat_max=0.00001)
#print(result)

true_ans = np.exp(time)
error = result - true_ans
print(error)
print(np.average(error))
# HAS REDUCING ACCURACY AS ESTIMATES DIVERGE FROM INITIAL POINT

'''
WORKS WITH 2 TIMES (time = [0, 1])

seems to screw up after first guess'''
