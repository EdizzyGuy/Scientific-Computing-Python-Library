# FIND A BETTER NAME FOR THIS FILE
# ADD AN EXTRA METHOD
import numpy as np
import time

def hopf_bifurcation(t, U, beta, sigma):
    u1 = U[0]
    u2 = U[1]

    u1_dot = beta*u1 - u2 + sigma *u1 *(u1*u1 + u2*u2)
    u2_dot = u1 + beta*u2 + sigma *u2 *(u1*u1 + u2*u2)
    U_dot = np.array([u1_dot, u2_dot])
    return U_dot

def hopf_extended(t, U, beta=2, sigma=-1):
    t = None

    U_dot = np.zeros(3)
    U_dot[:-1] = hopf_bifurcation(t, U[:-1], beta, sigma)
    U_dot[-1] = - U[-1]

    return U_dot

# NOW WORKING WITH PARAMETERS
# CHECK WITH PRED-PREY EQ


def euler_step(t, dx_dt, init_point, step_size, args):
    """ Will implement an euler step given the derivative of a system of equations (dx_dt)
    an initial point and a step size """
    increment = dx_dt(t, init_point, *args) * step_size

    new_point = init_point + increment
    new_time = t + step_size
    return new_point, new_time


def RK4_step(t, dx_dt, init_point, step_size, args):
    """ Will implement an increment in the independent variable of size step_size and find
    new value of dependent variable based on derivative of a system of equations (dx_dt)
    using the classic Runge-Kutta """
# initialise values of k
    k1 = dx_dt(t, init_point, *args)
    k2 = dx_dt(t + step_size/2, init_point + k1* step_size/2, *args)
    k3 = dx_dt(t + step_size/2, init_point + k2 *step_size/2, *args)
    k4 = dx_dt(t + step_size, init_point + k3 *step_size, *args)

    new_point = init_point + step_size/6 *(k1 +2*k2 + 2*k3 + k4)
    new_time = t + step_size
    return new_point, new_time


def solve_to(dx_dt, init_point, deltat_max=0.001, time_interval=[0, 1], method='RK4', args=()):
    """ Function will use a valid method to solve a system of equations characterised by dx_dt
    with a maximum step size between subsequent x being deltat_max.
    Will solve ODE from time_interval[START, FINISH], where start is taken as argument in case the dynamical
    system is implicitly time dependent"""

# MAKE USER UNABLE TO ENTER ANY OTHER METHOD THAN A VALID ONE
# TEST test to check if steps sum to time interval
    valid_methods = ['Euler', 'RK4']
    assert method in valid_methods, '1 step integration method supplied is not valid'

    current_step = init_point
    current_time = time_interval[0]

    match method:
        case 'Euler':
            while time_interval[1] - current_time > deltat_max:
                next_step, next_time = euler_step(current_time, dx_dt, current_step, step_size=deltat_max, 
                                        args=args)

                current_time = next_time
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step, last_time = euler_step(current_time, dx_dt, current_step, step_size=min_step,
                                    args=args)

        case 'RK4':
            while time_interval[1] - current_time > deltat_max:
                next_step, next_time = RK4_step(current_time, dx_dt, current_step, step_size=deltat_max,
                                                args=args)

                current_time = next_time
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step, last_time = RK4_step(current_time, dx_dt, current_step, step_size=min_step,
                                            args=args)
# NUMERICAL ISSUE : time does not add up to what it is meant to be exactly due to rounding errors
    return last_step, last_time


def solve_ode(dx_dt, init_cond, solve_for=np.linspace(0,10,100), deltat_max=0.001, method='RK4', args=()):
    """ This function will solve a system of differential equations characterised by dx_dt, with a given initial
    condition. Can use either 'RK4' or 'Euler' as a method of integration, and will implement these methods with the
    time step defined as deltat_max. Function will solve for all values of time inside the array 'solve_for'
    Returns an array of values of size m by n, where m is the number of time values to solve for and n is the
    dimensionality of the system of equations.
    When indexing the output first index will give a time point and second will give the corresponding variable of the
    system of equations at that time point. Time points are mapped from the solve_for array
    """
    x = np.zeros(shape=(len(solve_for), init_cond.size))
    x[0, :] = init_cond

    for i in range(len(solve_for) - 1):
        init_cond = x[i, :]
        time_interval = [solve_for[i], solve_for[i + 1]]
        x[i+1, :], current_time = solve_to(dx_dt, init_cond, deltat_max, time_interval, method, args)
    return x