# FIND A BETTER NAME FOR THIS FILE
# ADD AN EXTRA METHOD
import numpy as np
import time

# NOW WORKING WITH PARAMETERS
# CHECK WITH PRED-PREY EQ


def euler_step(t, dx_dt, initial_point, step_size, function_parameters):
    """ Will implement an euler step given the derivative of a system of equations (dx_dt)
    an initial point and a step size """
    increment = dx_dt(t, initial_point) * step_size

    new_point = initial_point + increment
    new_time = t + step_size
    return new_point, new_time


def RK4_step(t, dx_dt, initial_point, step_size, function_parameters):
    """ Will implement an increment in the independent variable of size step_size and find
    new value of dependent variable based on derivative of a system of equations (dx_dt)
    using the classic Runge-Kutta """
# initialise values of k
    k1 = dx_dt(t, initial_point, *function_parameters)
    k2 = dx_dt(t + step_size/2, initial_point + k1* step_size/2, *function_parameters)
    k3 = dx_dt(t + step_size/2, initial_point + k2 *step_size/2, *function_parameters)
    k4 = dx_dt(t + step_size, initial_point + k3 *step_size, *function_parameters)

    new_point = initial_point + step_size/6 *(k1 +2*k2 + 2*k3 + k4)
    new_time = t + step_size
    return new_point, new_time


def solve_to(dx_dt, initial_point, deltat_max=0.001, time_interval=[0, 1], method='RK4', function_parameters=()):
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
                next_step, next_time = euler_step(current_time, dx_dt, current_step, step_size=deltat_max, 
                                        function_parameters=function_parameters)

                current_time = next_time
                current_step = next_step
                i += 1
                if i == 10**5:
                    end = time.time()
                    elapsed = end - start
                    print(elapsed)
            min_step = time_interval[1] - current_time
            last_step, last_time = euler_step(current_time, dx_dt, current_step, step_size=min_step,
                                    function_parameters=function_parameters)

        case 'RK4':
            while time_interval[1] - current_time > deltat_max:
                next_step, next_time = RK4_step(current_time, dx_dt, current_step, step_size=deltat_max,
                                                function_parameters=function_parameters)

                current_time = next_time
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step, last_time = RK4_step(current_time, dx_dt, current_step, step_size=min_step,
                                            function_parameters=function_parameters)
# NUMERICAL ISSUE : time does not add up to what it is meant to be exactly due to rounding errors
    return last_step, last_time


def solve_ode(dx_dt, initial_condition, solve_for, deltat_max=0.001, method='RK4', function_parameters=()):
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
        initial_condition = x[i, :]
        time_interval = [solve_for[i], solve_for[i + 1]]
        x[i+1, :], current_time = solve_to(dx_dt, initial_condition, deltat_max, time_interval, method, function_parameters)
        if current_time != solve_for[i+1]:
            print('error')
    return x