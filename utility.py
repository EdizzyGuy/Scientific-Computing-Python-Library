import matplotlib.pyplot as plt
import numpy as np
from ode import solve_ode
from scipy.integrate import solve_ivp
# MAKE THIS WORK WITH NUMPY SOLVER


def get_phase_portrait(dXdt, init_cond, solve_for, portrait_variables=(0,1), solver=solve_ode, deltat_max=np.inf, method='RK4',
                        time_first=True, solver_args=dict(), t=None, title=None, xlabel=None, ylabel=None, args=()):
    """will solve a system of ODE's and plot their phase portrait. ONLY WORKS IN 2D
    TEST : PRED-PREY"""
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
    # plot phase portrait
    plt.plot(x, y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=8)
    plt.show()
    return path
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


