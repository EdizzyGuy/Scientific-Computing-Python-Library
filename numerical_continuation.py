import sys
import os
origin = r'C:\\Users\\Ediz\\Documents\\uni stuff\\Scientific Computing\Assignment\\ld18821-emat30008'
sys.path.append(origin)

import numpy as np
import time
from ode import solve_ode
from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve


PI = np.pi
solver_args={'deltat_max' : 0.01}
init_cond = np.array([1,1])
#np.random.normal(size=2) * 9
period_guess = np.random.uniform(low=5, high=25, size=1)
init_hopf = np.concatenate((init_cond, period_guess))

init_cond = np.array([1,1,1])
init_hopf_ext = np.concatenate((init_cond, period_guess))

'''
results = continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters
    vary_par=0,  # the parameter to vary
    step_size=0.1,  # the size of the steps to take
    max_steps=100,  # the number of steps to take
    discretisation=shooting,  # the discretisation to use
    solver=scipy.optimize.fsolve)

for cubic (simple root finding)
discretisation=lambda x: x

orig args : (dXdt, init_guess, ode_solver=solve_ode, root_solver=None, root_methods=('hybr', 'lm'), root_args=None,
                      deltat_max=np.inf, int_method='RK4', time_first=True, solver_args=dict(),
                      t=None, find_period=True, phase_condition=None, args=())
'''

def shooting(dXdt, explicit_time_dep=False, phase_condition=None, ode_solver=solve_ivp, solver_args=dict(), 
        time_first=True, args=()):
    ''' constructs the shooting root finding problem and outputs a function to be passed into
    root finding method in order to obtain limit cycles

    assume using solve_ivp due to superior performance (probably due to its variable step size)
    set exp time dep to false because most of our examples have no exp time dep
    time_first set as True by default so that I dont have to change it to test solve_ode

    Args:
        dXdt (callable) : derivative wrt time of the system
        explicit_time_dep (boolean) : should be true if the system of equations displays explicit time dep. Used in shooting.
        phase_condition (callable) : User defined phase condition in case that the default one fails
        ode_solver (callable) : Any valid function that can solve a system of ordinary differential equations
        solver_args (tuple) : positional arguements to be passed to the ode solver
        time_first (boolean)    : Should be true if ode solver outputs array such that timeis indexed first
        args (tuple) : positional arguements to be passed to dXdt
    Returns:
        root_finding_problem (callable) : root finding problem as defined in numerical shooting
    '''
    # intialise deltat_max if solve_ode
    if (ode_solver == solve_ode) and ('deltat_max' not in solver_args):
        solver_args.update({'deltat_max' : 0.01})
    
    # define function that minimises X(T) - X(0) wrt T 
    if ode_solver == solve_ivp:
    # special case for solve_ivp since it ouputs object
        def G(X, T):
            sol = ode_solver(dXdt, (0, T), X, **solver_args, args=args)
            trajectory_T = sol.y
            terminal_T = trajectory_T[:, -1]
            G = X - terminal_T
            return G
    elif time_first is True:
        # for use with my solver
        # but will work with any time first generic ode solver as long as first 3 args are same
        def G(X, T):
            trajectory_T = ode_solver(dXdt, X, [0, T], **solver_args, args=args)
            terminal_T = trajectory_T[-1,:]
            G = X - terminal_T
            return G
    else:
        # for use with a generic ode solver
        # assumes structure of scipy.integrate.odeint
        def G(X, T):
            trajectory_T = ode_solver(dXdt, X, [0, T], **solver_args, args=args)
            terminal_T = trajectory_T[:,-1]
            G = X - terminal_T
            return G
    
    # define phase condition
    if explicit_time_dep is True:
        phase_condition = lambda X : 0
    elif phase_condition is None:
        def phase_condition(X):
            t = None
            X_dot = dXdt(t, X, *args)
            return X_dot[0]  # will only work for the variable that is oscillating

    def root_finding_problem(init_guess):
        X = init_guess[:-1] # state
        T = init_guess[-1] # period
        sol = np.hstack((G(X, T), phase_condition(X)))
        return sol

    return root_finding_problem

#(dXdt, init_guess, ode_solver=solve_ode, root_solver=None, root_methods=('hybr', 'lm'), root_args=None,
    #                  deltat_max=np.inf, int_method='RK4', time_first=True, solver_args=dict(),
    #                  t=None, find_period=True, phase_condition=None, args=())
def find_limit_cycles(dXdt, init_guess, ode_solver=solve_ivp, root_solver=None, discretization=shooting,
        root_methods=('hybr', 'lm'), time_first=True, explicit_time_dep=False, 
        discr_args=dict(), discr_kwargs=dict(), root_args=None, 
        solver_args=dict(), args=()):
    ''' attempts to find potentially stable limit cycles in the system defined by dXdt. This is achieved by constructing the
    appropriate root finding problem as defined in shooting by default, but other root based methods are applicable.

    Args:
        dXdt (callable) : derivative wrt time of the system
        ode_solver (callable) : Any valid function that can solve a system of ordinary differential equations
        root_solver (callable) : function to solve roots
        discretization (callable) : Function used to discretize the limit cycle
        root_methods (iterable) : in case of using scipy.optimize.roots will iterate through this iterable of methods in order
                                    until a solution is reached or the end of the list is reached.
        time_first (boolean)    : Should be true if ode solver outputs array such that timeis indexed first
        explicit_time_dep (boolean) : should be true if the system of equations displays explicit time dep. Used in shooting.
        discr_args (tuple) : positional arguements to be passed to the discretization function
        discr_kwargs (dict) : key word arguements to be supllied to the discretization function
        root_args (tuple) : positional arguements to be passed to the root solver
        solver_args (tuple) : positional arguements to be passed to the ode solver
    Returns:
        roots (np.ndarray) : roots of the root finding problem (state vector of position on cycle, period of cycle)
        None               : returned in case that no roots were found
    orig args : (deltat_max=np.inf, int_method='RK4', 
                      t=None, phase_condition=None)'''
    # default to no extra arguements passed to root solver
    if root_args is None: 
        if root_solver is None:
            root_args = [dict() for i in range(len(root_methods))] 
        else:
            # TEST PASSING NON CALLABLE FUNCTION
            assert callable(root_solver), 'root solver must be a callable function'
            root_args = dict()
    assert len(root_methods) == len(root_args), 'Each root method must be given a corresponding arguement'
    
  # seperating ode_solver and time_first and exp time dep from discr_args because these arguements will affect other discretization techniques
    discr_func = discretization(dXdt, explicit_time_dep, ode_solver=ode_solver, time_first=time_first, solver_args=solver_args, *discr_args, **discr_kwargs, args=args)
    
    # special case for scipy root since returns object
    if root_solver == None:
        for i, method in enumerate(root_methods):
            sol = root(discr_func, init_guess, method=method, options=root_args[i])
            if sol.success:
                return sol.x
        print('root solver has failed to converge to a valid solution... try a different solver/method')
        return None
    # for a generic root solver
    else:
        roots = root_solver(discr_func, init_guess, **root_args)
        validity = discr_func(roots)
        if np.all(np.isclose(validity, 0, atol=1e-04)):
            return roots
        else:
            print('root solver has failed to converge to a valid solution... try a different solver')
            return None


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


