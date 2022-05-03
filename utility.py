import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from ode import solve_ode
from scipy.integrate import solve_ivp
import time
# MAKE THIS WORK WITH NUMPY SOLVER


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

def display_dynamic_solution(u, x, t, title, comp_time=10):
    
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


def mean_square_error(A, B, ax=None):
    mse = (np.square(A - B)).mean(ax)
    return mse

def abs_error(A, B):
    error = A - B
    abs_error = np.sum(np.abs(error))
    return abs_error

def isInteger(N):
    X = int(N)
 
    diff = N - X
    if np.isclose(diff, 0):
        return True
         
    return False

def time_function(func, count=1, args=tuple(), kwargs=dict()):
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


