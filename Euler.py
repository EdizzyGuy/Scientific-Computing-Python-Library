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
    """ Function will use Euler's method to solve a system of equations characterised by x
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
                i += 1
                if i == 10 ** 5:
                    end = time.time()
                    elapsed = end - start
                    print(elapsed)
            min_step = time_interval[1] - current_time
            last_step = RK4_step(dx_dt, current_step, step_size=min_step)
# NUMERICAL ISSUE : time does not add up to what it is meant to be exactly due to rounding errors
    return last_step


def solve_ode(dx_dt, initial_condition, time, deltat_max=0.001, method='RK4'):
    x = np.zeros(shape=(len(time), initial_condition.size))
    x[0, :] = initial_condition

    for i in range(len(time)-1):
        x[i+1, :] = solve_to(dx_dt, x[i, :], deltat_max, [time[i], time[i+1]], method=method)
    return x


'''
def solve_to(dx_dt, initial_condition, deltat_max=0.001, time_interval=[0, 1]):
    pass
    """ Initial condition must be given as a numpy array """
    time = np.arange(time_interval[0], time_interval[1], step=deltat_max)
    time = np.append(time, time_interval[1])
    x = np.zeros(shape=(initial_condition.size, len(time)))
    x[:, 0] = initial_condition  # row major form therefore row first

    deltat = time[1] - time[0]
    next_euler = euler_step(dx_dt, initial_condition, deltat)
    for i in range(len(time)):
        deltat = time[i+1] - time[i]
        next_euler = euler_step(dx_dt, x[:, i], deltat)
        x[:, i+1] = next_euler

    return x, time '''

