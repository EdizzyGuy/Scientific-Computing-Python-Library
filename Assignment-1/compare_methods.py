import sys
import os
path = sys.path[0]
parent = os.path.dirname(path)
sys.path.append(parent)

from rk4_euler_graph import errors_RK4, errors_euler, time_steps, dx_dt, initial_condition
import ode
from utility import *
import numpy as np
import time

if __name__ =='__main__':
    best_tolerance = np.inf
    eul_indx = None
    rk4_indx = None
    for i, error_eul in enumerate(np.abs(errors_euler)):
        for j, error_RK4 in enumerate(np.abs(errors_RK4)):
            if error_eul*0.95 < error_RK4 < error_eul*1.05:  # if rk4 error within +/- 5% of euler error
                tolerance = np.abs((error_RK4/ error_eul)*100 - 100)

                print(f'euler error index {i} and rk4 error index {j} have error rates within 10% of each other')
                print(f'euler error is {error_eul}')
                print(f'rk4 error is {error_RK4}')
                print(f'rk4 error is within {tolerance:.2}% of euler')
                print()

                if tolerance < best_tolerance:
                    best_tolerance = tolerance
                    eul_indx = i
                    rk4_indx = j

    eul_time_step = time_steps[eul_indx]
    rk4_time_step = time_steps[rk4_indx]
    print(f'step sizes {eul_time_step} and {rk4_time_step} give very similar errors for euler '
        f'and runge-kutta respectively')

    # find out how long each method takes with this step size
    solve_between = np.array([0, 1])

    eul_time,eul_std = time_function(solve_ode, 100, args=(dx_dt, initial_condition, solve_between), kwargs={'deltat_max':eul_time_step, 'method':ode.euler_step})
    rk4_time, rk4_std = time_function(solve_ode, 1000, args=(dx_dt, initial_condition, solve_between), kwargs={'deltat_max':rk4_time_step, 'method':ode.RK4_step})

    print(f'Euler time          : {eul_time}')
    print(f'Runge-kutta time    : {rk4_time}')

    quickest_method = None
    if eul_time < rk4_time:
        quickest_method = 'Euler is'
    elif rk4_time == eul_time:
        quickest_method = 'Both methods are'
    else:
        quickest_method = 'Runge-Kutta is'

    print()
    print(f'{quickest_method} is the quickest method for solve_ode function')
