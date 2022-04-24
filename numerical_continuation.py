import numpy as np
from ode import solve_ode
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#TODO add option to set step size for ode solver and to choose different solver
def find_limit_cycles(init_guess, dXdt, solver=solve_ode, deltat_max=np.inf, method='RK4', time_first=True, solver_args=dict(),
                      t=None, find_period=True, phase_condition=None, args=()):
    """ will find limit cycles - can input any requested phase_condition
    init guess must have state variables first then guess of period if period is known put known period in
    handles other functions
    handles different phase cond

    TESTS : test on higher dimensional question
            test on solution with no phase condition
            test different phase condition
            test solve_ivp func and other func with state first output
            test different args AND passing args as dict
            
    ADD : options for the numerical integrator, options for the numerical root finder,
    
    assumes parameter structure of an arbitrary solver
    thought about adding t dependence to phase cond but I dont think its possible because
        fsolve will constantly change start position and therefore the time of initialisation (if function is time variant)
        would be a different problem where t must be included as state variable
        
    WHY is my ode solver taking so much longer than theirs"""
# dont want to use if else inside function definitions
    if solver == solve_ivp:
        # REMOVE THIS ONCE RK45 IMPLEMENTED IN SOLVE_ODE
        if method == 'RK4':
            method = 'RK45'
            
        def G(dXdt, init_guess, args):
            X = init_guess[:-1]
            T = init_guess[-1]
            sol = solver(dXdt, (0, T), X, method, **solver_args, args=args)
            trajectory_T = sol.y
            terminal_T = trajectory_T[:, -1]
            G = X - terminal_T
            return G

    elif time_first is not True:
        def G(dXdt, init_guess, args):
            X = init_guess[:-1]
            T = init_guess[-1]
            trajectory_T = solver(dXdt, X, solve_for=[0, T], args=args)
            terminal_T = trajectory_T[:,-1]
            G = X - terminal_T
            return G

    else:
        if deltat_max == np.inf:
            deltat_max = 0.01

        def G(dXdt, init_guess, args):
            X = init_guess[:-1]
            T = init_guess[-1]
            trajectory_T = solver(dXdt, X, [0, T], deltat_max, method, **solver_args, function_parameters=args)
            terminal_T = trajectory_T[-1,:]
            G = X - terminal_T
            return G


    if find_period is False:
        phase_condition = lambda X, dXdt, args : 0
    elif (find_period is True) and (phase_condition is None):

        def phase_condition(X, dXdt, args):
            t = None
            X_dot = dXdt(t, X, *args)
            return X_dot[0]  # will only work for the variable that is oscillating


    def root_finding_problem(init_guess, G, phase_condition, t, dXdt,  args):
        sol = np.zeros(init_guess.shape)
        sol[:-1] = G(dXdt, init_guess, args)
        sol[-1] = phase_condition(init_guess[:-1], dXdt, args)
        return sol    

    roots = fsolve(root_finding_problem, init_guess, args=(G, phase_condition, t,dXdt, args))
    return roots
