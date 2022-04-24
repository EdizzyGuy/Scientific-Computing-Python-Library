import matplotlib.pyplot as plt
import numpy as np
from ode import solve_ode
from scipy.integrate import solve_ivp

# MAKE THIS WORK WITH NUMPY SOLVER
def get_phase_portrait(initial_condition, dXdt, solve_for, portrait_variables=(0,1), solver=solve_ode, deltat_max=np.inf, method='RK4',
                        time_first=True, solver_args=dict(), t=None, title=None, xlabel=None, ylabel=None, args=()):
    """will solve a system of ODE's and plot their phase portrait. ONLY WORKS IN 2D
    TEST : PRED-PREY"""
    if solver == solve_ivp:
    # REMOVE THIS ONCE RK45 IMPLEMENTED IN SOLVE_ODE
        if method == 'RK4':
            method = 'RK45'
    elif time_first is not True:
        # only input 
        solution = solver(dXdt, initial_condition, **solver_args, args=args)
    else:
        solution = solver(dXdt, initial_condition, solve_for, deltat_max, method, **solver_args, args=args)
    

    x = solution[:, portrait_variables[0]]
    y = solution[:, portrait_variables[1]]
    # plot phase portrait
    plt.plot(x, y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=8)
    plt.show()
    return solution
    # STABLE LIMIT CYCLE -> it has been shown that solutions collapse to a stable equilibrium at b =0.27


def mean_square_error(A, B, ax=None):
    mse = (np.square(A - B)).mean(ax)
    return mse

def isInteger(N):
    X = int(N)
 
    diff = N - X
    if np.isclose(diff, 0):
        return True
         
    return False

def test():
    print('yes')
    return