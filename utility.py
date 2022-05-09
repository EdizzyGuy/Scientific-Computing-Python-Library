import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from ode import solve_ode
from scipy.integrate import solve_ivp
import time
# MAKE THIS WORK WITH NUMPY SOLVER

#TD
def mean_square_error(A, B, ax=None):
    """ Returns the mean of the square error between elements in A and B along a defined axis. Axis is defaulted
    to 'None', i.e. will return mean of square error between ALL elements. A and B must be the same size.
    
    Args:
        A (np.ndarray) : numpy array to compare/ be compared with B.
        B (np.ndarray) : numpy array to compare/ be compared with A.
        ax (int)       : axis along to take mean of the square error.
        
    Returns:
        mse            : Mean of the square error between ALL elements in A and B.
    """
    mse = (np.square(A - B)).mean(ax)
    return mse

#TD
def abs_error(A, B):
    """ Returns summation of absolute error between arrays A and B. A and B must be the same size.
    
    Args:
        A (np.ndarray)    : numpy array to compare/ be compared with B
        B (np.ndarray)    : numpy array to compare/ be compared with A
        
    Returns:
        abs_error (float) : Summation of absolute error between elements in A and B
    """
    error = A - B
    abs_error = np.sum(np.abs(error))
    return abs_error

#TD
def mean_rel_error(A, B):
    """ Returns the average value of A / B (performed elementwise) and is used in cases where values in B and A
    are too close to 0 for absolute error or mean squared error to carry meaning. . A and B must be the same size.
    
    Args: 
        A (np.ndarray)    : Numpy array to compare with, and find relative error against, referance array B.
        B (np.ndarray)    : Referance array that is being compared with A
        
    Returns:
        rel_error (float) : Average value of the elementwise divison of A and B
    """
    division_array = np.divide(A, B)

    valid_terms = np.count_nonzero(~np.isnan(division_array))  # number of non Nan values
    valid_terms -= np.count_nonzero(division_array==0) + np.count_nonzero(division_array==np.inf)
    division_array = np.nan_to_num(division_array, copy=True, nan=0.0, posinf=None, neginf=None)  #replace all nan with 0
    
    rel_error = np.sum(np.abs(division_array))
    mean_rel_error = rel_error / valid_terms  # devide by num of terms that werent nan to retain accuracy

    return mean_rel_error

#T
def get_phase_portrait(dXdt, init_cond, solve_for, solver=solve_ode, title=None, xlabel=None, ylabel=None, portrait_variables=(0,1), title_size=11, centre_spines=True,
                       deltat_max=np.inf, method='RK4', time_first=True, solver_args=dict(), t=None, args=()):
    """will solve a system of ODE's and plot their phase portrait. ONLY WORKS IN 2D
    TEST : PRED-PREY"""
    if xlabel is None and ylabel is None:
        xlabel=r'$\mathbf{u_1}$'
        ylabel=r'$\mathbf{u_2}$'


    if solver == solve_ivp:
    # REMOVE THIS ONCE RK45 IMPLEMENTED IN SOLVE_ODE
        if method == 'RK4':
            method = 'RK45'
        sol = solver(dXdt, (0, solve_for[-1]), init_cond, method, solve_for, max_step=deltat_max, **solver_args, args=args)
        path = sol.y.T
    elif time_first is not True:
        # only include dxdt and init_cond as input to allow for flexibility in handleable args
        path = solver(dXdt, init_cond, **solver_args, args=args)
    else:
        if deltat_max == np.inf:
            deltat_max = 1e-03
        path = solver(dXdt, init_cond, solve_for, deltat_max, method, **solver_args, args=args)
    

    x = path[:, portrait_variables[0]]
    y = path[:, portrait_variables[1]]

    plt.figure(figsize=(8,6), dpi=100)
    ax = plt.subplot(111)
    # plot phase portrait
    plt.plot(x, y, color='blue')

    if centre_spines:
        plt.axhline(color='black', lw=1)
        plt.axvline(color='black', lw=1)

    ax.set_xlabel(xlabel, va='center', ha='center')
    ax.set_ylabel(ylabel, rotation=0, va='bottom', ha='center')
    ax.set_title(title, fontsize=title_size)
    plt.grid()
    plt.show()
    
    return path
    # STABLE LIMIT CYCLE -> it has been shown that solutions collapse to a stable equilibrium at b =0.27

def display_dynamic_solution(u, x, t, title=None, comp_time=10):
    
    x_upper = x.max()
    x_lower = x.min()
    u_upper = u[0].max()
    u_lower = u[0].min()

    fig = plt.figure(figsize=(6,6), facecolor='white')
    fps = len(t) / comp_time
    period = 1000 / fps  # given in millisecs

    def update(frame):
        plt.clf()
        plt.plot(x,u[frame],'r-')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel(f'u(x,{t[frame]:.4f})')
        plt.xlim(x_lower, x_upper)
        plt.ylim(u_lower, u_upper)
        plt.show()

    animation = FuncAnimation(fig, update, interval=period, frames=len(t))
    plt.show() 

#TD
def isInteger(N):
    """ Function that will return True if the inputted value is an integer, and false if not
    
    Args:
        N (float) : Numerical object to evaluate
        
    Returns:
        (bool)    : Result of the evaluation"""
    X = int(N)
 
    diff = N - X
    if np.isclose(diff, 0):
        return True
         
    return False

#D
def time_function(func, count=1, args=tuple(), kwargs=dict()):
    """ Runs the inputted function {count} number of times and returns the average and the standard deviation
    of these durations. Can take arguements and key word arguements to the function.
    
    Args:
        func (callable)     : function to time.
        count (int)         : number of times to time function.
        args (tuple)        : arguements to be passed to the func, in order
        kwargs (dictionary) : Key value pairs of key-word to arguement, to be passed into func
        
    Returns:
        avg (float)         : Average duration taken to evaluate function
        std (float)         : Standard deviation for said averages
    """
    times = np.zeros(count)
    for i in range(count):
        start = time.time()
        sol = func(*args, **kwargs)
        end = time.time()
        times[i] = end-start
    avg = np.average(times)
    std = np.std(times)
    if count == 1:
        std = np.inf
    return avg, std


