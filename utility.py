import ode
import matplotlib.pyplot as plt
import numpy as np

def get_phase_portrait(dXdt, initial_condition, solve_for, title=None, xlabel=None, ylabel=None, portrait_variables=(0,1), 
                            deltat_max=0.001, method="RK4", function_parameters=()):
    """will solve a system of ODE's and plot their phase portrait. ONLY WORKS IN 2D
    TEST : PRED-PREY"""
    solution = ode.solve_ode(dXdt, initial_condition, solve_for, deltat_max, method, function_parameters)
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

def test():
    print('yes')
    return