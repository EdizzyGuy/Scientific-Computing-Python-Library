import ode
import numpy as np
from scipy.optimize import fsolve

def find_limit_cycles(init_guess, dXdt, find_period=True, phase_condition=None, args=()):
    """ will find limit cycles - can input any requested phase_condition
    init guess must have state variables first then guess of period if period is known put known period in
    TESTS : test on higher dimensional question
            test on solution with known period
            
    ADD : options for the numerical integrator, options for the numerical root finder,"""
    t = None
    if find_period is False:
        phase_condition = lambda X, dXdt, args : 0
    elif (find_period is True) and (phase_condition is None):

        def phase_condition(t, X, dXdt, args):
            X_dot = dXdt(t, X, *args)
            return X_dot[0]  # will only work for the variable that is oscillating

    def G(dXdt, init_guess, args):
        X = init_guess[:-1]
        T = init_guess[-1]
        trajectory_T = ode.solve_ode(dXdt, X, solve_for=[0, T], function_parameters=args)
        terminal_T = trajectory_T[-1,:]
        G = X - terminal_T
        return G

    def root_finding_problem(init_guess, G, phase_condition, t, dXdt,  args):
        sol = np.zeros(init_guess.shape)
        sol[:-1] = G(dXdt, init_guess, args)
        sol[-1] = phase_condition(t, init_guess[:-1], dXdt, args)
        return sol    

    roots = fsolve(root_finding_problem, init_guess, args=(G, phase_condition, t,dXdt, args))
    return roots