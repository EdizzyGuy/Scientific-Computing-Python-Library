# ADD VARIABLE STEP SIZE
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


def euler_step(t, dx_dt, init_point, step_size, args, kwargs):
    """ Will implement a forward euler step given the derivative of a system of equations (dx_dt)
    an initial point and a step size 
    
    Args:
        t (float)               : time variable, is required
        dx_dt (callable)        : derivative of system wrt time
        init_point (np.ndarray) : initial point to implement the euler step on
        step_size (float)       : magnitude of the euler step (in time)
        args (tuple)            : optional arguements to be passed to dx_dt
        kwargs (dict)           : keyword arguements to be passed to dx_dt
        
    Returns:
        new_point (np.ndarray)  : initial point after the application of an euler step
        new_time (float)        : new value of time after the euler step
    """
    increment = dx_dt(t, init_point, *args, **kwargs) * step_size

    new_point = init_point + increment
    new_time = t + step_size
    return new_point, new_time


def RK4_step(t, dx_dt, init_point, step_size, args, kwargs):
    """ Will implement an increment in the independent variable of size step_size and find
    new value of dependent variable based on derivative of a system of equations (dx_dt)
    using the classic Runge-Kutta
    
    Args:
        t (float)               : time variable, is required
        dx_dt (callable)        : derivative of system wrt time
        init_point (np.ndarray) : initial point to implement the euler step on
        step_size (float)       : magnitude of the euler step (in time)
        args (tuple)            : positional arguements to be passed to dx_dt
        kwargs (dict)           : keyword arguements to be passed to dx_dt
        
    Returns:
        new_point (np.ndarray)  : initial point after the application of an euler step
        new_time (float)        : new value of time after the euler step
    """
# initialise values of k
    k1 = dx_dt(t, init_point, *args)
    k2 = dx_dt(t + step_size/2, init_point + k1* step_size/2, *args, **kwargs)
    k3 = dx_dt(t + step_size/2, init_point + k2 *step_size/2, *args, **kwargs)
    k4 = dx_dt(t + step_size, init_point + k3 *step_size, *args, **kwargs)

    new_point = init_point + step_size/6 *(k1 +2*k2 + 2*k3 + k4)
    new_time = t + step_size
    return new_point, new_time

def RK45_step(t, dx_dt, init_point, step_size, args, kwargs):
    """ Will implement an increment in the independent variable of size step_size and find
    new value of dependent variable based on derivative of a system of equations (dx_dt)
    using the 5th order Runge-Kutta method.
    
    Args:
        t (float)               : time variable, is required
        dx_dt (callable)        : derivative of system wrt time
        init_point (np.ndarray) : initial point to implement the euler step on
        step_size (float)       : magnitude of the euler step (in time)
        args (tuple)            : positional arguements to be passed to dx_dt
        kwargs (dict)           : keyword arguements to be passed to dx_dt
        
    Returns:
        new_point (np.ndarray)  : initial point after the application of an euler step
        new_time (float)        : new value of time after the euler step
    """
    k1 = step_size * dx_dt(t, init_point, *args, **kwargs)
    k2 = step_size* dx_dt(t + step_size/2, init_point + k1/2, *args, **kwargs)
    k3 = step_size * dx_dt(t+step_size/2, init_point+(k1+k2)/4, *args, **kwargs)
    k4 = step_size * dx_dt(t + step_size, init_point-k2+2*k3, *args, **kwargs)
    k5 = step_size * dx_dt(t+step_size/3,init_point+7/27*k1 +10/17*k2 +1/27*k4, *args, **kwargs)
    k6 = step_size *dx_dt(t+1/5*step_size,init_point +28/625*k1 -1/5*k2 +546/625*k3 +54/625*k4 -378/625*k5, *args, **kwargs)

    new_point = init_point +1/24*k1 +5/48*k4 +27/56*k5 +125/336*k6
    new_time = t +step_size
    return new_point, new_time

def solve_to(dx_dt, init_point, deltat_max=0.001, time_interval=[0, 1], method='RK4', args=(), kwargs=dict()):
    """ Function will use a valid method to solve a system of equations characterised by dx_dt
    with a maximum step size between subsequent x being deltat_max.
    Will solve ODE from time_interval[START, FINISH], where start is taken as argument in case the dynamical
    system is implicitly time dependent
    
    Args:
        dx_dt (callable)            : derivative of system wrt time
        init_point (np.ndarray)     : initial point to implement the euler step on
        deltat_max (float)          : maximum allowed step size, keep small for improved accuracy
        time_interval (list-like)   : time interval in which to integrate ODE
        method (str)                : one step integration method
        args (tuple)                : optional arguements to be passed to dx_dt
        kwargs (dict)               : keyword arguements to be passed to dx_dt
        
    Returns:
        last_step (np.ndarray)      : state of the integrated ODE at the end of the time interval
        last_time (float)           : value of time after integrating ODE between time_interval
    """

# MAKE USER UNABLE TO ENTER ANY OTHER METHOD THAN A VALID ONE
# TEST test to check if steps sum to time interval
    valid_methods = ['Euler', 'RK4', 'RK45']
    assert method in valid_methods, '1 step integration method supplied is not valid'

    current_step = init_point
    current_time = time_interval[0]

    match method:
        case 'Euler':
            while time_interval[1] - current_time > deltat_max:
                next_step, next_time = euler_step(current_time, dx_dt, current_step, step_size=deltat_max, 
                                        args=args, kwargs=kwargs)

                current_time = next_time
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step, last_time = euler_step(current_time, dx_dt, current_step, step_size=min_step,
                                    args=args, kwargs=kwargs)

        case 'RK4':
            while time_interval[1] - current_time > deltat_max:
                next_step, next_time = RK4_step(current_time, dx_dt, current_step, step_size=deltat_max,
                                                args=args, kwargs=kwargs)

                current_time = next_time
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step, last_time = RK4_step(current_time, dx_dt, current_step, step_size=min_step,
                                            args=args, kwargs=kwargs)

        case 'RK45':
            while time_interval[1] - current_time > deltat_max:
                next_step, next_time = RK45_step(current_time, dx_dt, current_step, step_size=deltat_max,
                                                args=args, kwargs=kwargs)

                current_time = next_time
                current_step = next_step
            min_step = time_interval[1] - current_time
            last_step, last_time = RK45_step(current_time, dx_dt, current_step, step_size=min_step,
                                            args=args, kwargs=kwargs)


# NUMERICAL ISSUE : time does not add up to what it is meant to be exactly due to rounding errors
    return last_step, last_time


def solve_ode(dx_dt, init_cond, solve_for=np.linspace(0,10,100), deltat_max=0.001, method='RK4', args=(), kwargs=dict()):
    """ This function will solve a system of differential equations characterised by dx_dt, with a given initial
    condition. Can use either 'RK4' or 'Euler' as a method of integration, and will implement these methods with the
    time step defined as deltat_max. Function will solve for all values of time inside the array 'solve_for'
    Returns an array of values of size m by n, where m is the number of time values to solve for and n is the
    dimensionality of the system of equations.
    When indexing the output first index will give a time point and second will give the corresponding variable of the
    system of equations at that time point. Time points are mapped from the solve_for array.

    Args:
        dx_dt (callable)            : derivative of system wrt time
        init_cond (np.ndarray)      : state of the system at time 0
        solve_for (np.ndarray)      : array of values to return the intgrated ODE at
        deltat_max (float)          : maximum allowed step size for the one step intgration, decrease for increased accuracy
        method (str)                : method of one step intgration, choose 'Euler' or 'RK4'
        args (tuple)                : positional arguements to be passed to dx_dt
        kwargs (dict)               : keyword arguements to be passed to dx_dt

    Returns:
        x (np.ndarray)              : State of the integrated system of ODE's at time points defined by solve_for
                                      First index will return state at time point defined by solve_for and the second 
                                      will return the state variable in question:
                                      e.g. for X=[x,y] and solve_for = [0,1] and a suitable initial condition and derivative function
                                      x[1,1] will give the value of y at time 1. 
    """
    x = np.zeros(shape=(len(solve_for), init_cond.size))
    x[0, :] = init_cond

    for i in range(len(solve_for) - 1):
        init_cond = x[i, :]
        time_interval = [solve_for[i], solve_for[i + 1]]
        x[i+1, :], current_time = solve_to(dx_dt, init_cond, deltat_max, time_interval, method, args, kwargs)
    return x