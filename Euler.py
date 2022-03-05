# FIND A BETTER NAME FOR THIS FILE
import numpy as np
import time


def euler_step(dx_dt, initial_point, step_size):
    """ Will implement an euler step given the derivative of a system of equations (dx_dt)
    an initial point and a step size """
    increment = dx_dt(initial_point) * step_size
    new_point = initial_point + increment
    return new_point


def RK4_step(dx_dt, initial_point, step_size):
    """ Will implement an increment in the independent variable of size step_size and find
    new value of dependent variable based on derivative of a system of equations (dx_dt)
    using the classic Runge-Kutta """
# initialise values of k
    k1 = dx_dt(initial_point)
    k2 = dx_dt(initial_point + step_size*k1/2)
    k3 = dx_dt(initial_point + step_size*k2/2)
    k4 = dx_dt(initial_point + step_size*k3)

    new_point = initial_point + step_size/6 *(k1 +2*k2 + 2*k3 + k4)
    return new_point


def solve_to(dx_dt, initial_point, deltat_max=0.001, time_interval=[0, 1], method='RK4'):
    """ Function will use a valid method to solve a system of equations characterised by dx_dt
    with a maximum step size between subsequent x being deltat_max.
    Will solve ODE from time_interval[START, FINISH], where start is taken as argument in case the dynamical
    system is implicitly time dependent"""

# MAKE USER UNABLE TO ENTER ANY OTHER METHOD THAN A VALID ONE
# TEST test to check if steps sum to time interval
    valid_methods = ['Euler', 'RK4']
    if method not in valid_methods:
        # WHAT ERROR SHOULD I RAISE
        # ALSO MAKE IT BREAK THE FUNCTION AND SUPERCEEDING FUNCTIONS
        pass

    current_step = initial_point
    current_time = time_interval[0]
    start = time.time()
    i = 0

    match method:
        case 'Euler':
            while time_interval[1] - current_time > deltat_max:
                next_step = euler_step(dx_dt, current_step, step_size=deltat_max)

                current_time += deltat_max
                current_step = next_step
                i += 1
                if i == 10**5:
                    end = time.time()
                    elapsed = end - start
                    print(elapsed)
            min_step = time_interval[1] - current_time
            last_step = euler_step(dx_dt, current_step, step_size=min_step)

        case 'RK4':
            while time_interval[1] - current_time > deltat_max:
                next_step = RK4_step(dx_dt, current_step, step_size=deltat_max)

                current_time += deltat_max
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step = RK4_step(dx_dt, current_step, step_size=min_step)
# NUMERICAL ISSUE : time does not add up to what it is meant to be exactly due to rounding errors
    return last_step


def solve_ode(dx_dt, initial_condition, solve_for, deltat_max=0.001, method='RK4'):
    """ This function will solve a system of differential equations characterised by dx_dt, with a given initial
    condition. Can use either 'RK4' or 'Euler' as a method of integration, and will implement these methods with the
    time step defined as deltat_max. Function will solve for all values of time inside the array 'solve_for'
    Returns an array of values of size m by n, where m is the number of time values to solve for and n is the
    dimensionality of the system of equations.
    When indexing the output first index will give a time point and second will give the corresponding variable of the
    system of equations at that time point. Time points are mapped from the solve_for array
    """
    x = np.zeros(shape=(len(solve_for), initial_condition.size))
    x[0, :] = initial_condition

    for i in range(len(solve_for) - 1):
        x[i+1, :] = solve_to(dx_dt, x[i, :], deltat_max, [solve_for[i], solve_for[i + 1]], method=method)
    return x

