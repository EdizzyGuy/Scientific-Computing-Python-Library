import numpy as np


def euler_step(dx_dt, initial_point, step_size):
    """ Will implement an euler step given the derivative of a system of equations (dx_dt)
    an initial point and a step size """
    increment = dx_dt(initial_point) * step_size
    new_point = initial_point + increment
    return new_point


def solve_to(dx_dt, initial_point, deltat_max=0.001, time_interval=[0, 1]):
    """ Function will use Euler's method to solve a system of equations characterised by x
    with a maximum step size between subsequent x being deltat_max.
    Will solve ODE from time_interval[START, FINISH], where start is taken as argument in case the dynamical
    system is implicitly time dependent"""

    num_steps = int(np.ceil((time_interval[1] - time_interval[0])/deltat_max))
    steps = [deltat_max for i in range(num_steps - 1)]
    steps.append(time_interval[1] - deltat_max*num_steps)

    current_euler = initial_point
    current_time = time_interval[0]
    for step in steps:
        next_euler = euler_step(dx_dt, current_euler, step_size=step)

        current_time += step
        current_euler = next_euler
    return current_euler


def solve_ode(dx_dt, initial_condition, time, deltat_max=0.001):
    x = np.zeros(shape=(initial_condition.size, len(time)))
    x[:, 0] = initial_condition

    for i in range(len(time)):
        x[:, i+1] = solve_to(dx_dt, x[:, i], deltat_max, [time[i], time[i+1]])
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

