import numpy as np
from ode import solve_ode
from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve

#TODO add option to set step size for ode solver and to choose different solver
def find_limit_cycles(dXdt, init_guess, ode_solver=solve_ode, root_solver=None, root_methods=('hybr', 'lm'), root_args=None,
                      deltat_max=np.inf, int_method='RK4', time_first=True, solver_args=dict(),
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
    # default to no extra arguements
    if root_args is None: 
        if root_solver is None:
            root_args = [dict() for i in range(len(root_methods))] 
        else:
            assert callable(root_solver), 'root solver must be a callable function'
            root_args = dict()
    assert len(root_methods) == len(root_args), 'Each root method must be given a corresponding arguement'

# dont want to use if else inside function definitions
    if ode_solver == solve_ivp:
        # REMOVE THIS ONCE RK45 IMPLEMENTED IN SOLVE_ODE
        if int_method == 'RK4':
            int_method = 'RK45'
            
        def G(init_guess):
            X = init_guess[:-1]
            T = init_guess[-1]
            sol = ode_solver(dXdt, (0, T), X, int_method, max_step=deltat_max,**solver_args, args=args)
            trajectory_T = sol.y
            terminal_T = trajectory_T[:, -1]
            G = X - terminal_T
            return G

    elif time_first is not True:
        def G(init_guess):
            X = init_guess[:-1]
            T = init_guess[-1]
            trajectory_T = ode_solver(dXdt, X, solve_for=[0, T], args=args)
            terminal_T = trajectory_T[:,-1]
            G = X - terminal_T
            return G

    else:
        if deltat_max == np.inf:
            deltat_max = 0.01

        def G(init_guess):
            X = init_guess[:-1]
            T = init_guess[-1]
            trajectory_T = ode_solver(dXdt, X, [0, T], deltat_max, int_method, **solver_args, args=args)
            terminal_T = trajectory_T[-1,:]
            g = X - terminal_T
            return g


    if find_period is False:
        phase_condition = lambda X : 0
    elif (find_period is True) and (phase_condition is None):

        def phase_condition(X):
            t = None
            X_dot = dXdt(t, X, *args)
            return X_dot[0]  # will only work for the variable that is oscillating


    def root_finding_problem(init_guess):
        X = init_guess[:-1]
        sol = np.hstack((G(init_guess), phase_condition(X)))
        return sol    

    if root_solver == None:
        for i, method in enumerate(root_methods):
            sol = root(root_finding_problem, init_guess, method=method, **root_args[i])
            if sol.success:
                return sol.x
        print('root solver has failed to converge to a valid solution... try a different solver/method')
        return None
    else:
        roots = root_solver(root_finding_problem, init_guess, **root_args)
        validity = root_finding_problem(roots)
        if np.all(np.isclose(validity, 0, atol=1e-04)):
            return roots
        else:
            print('root solver has failed to converge to a valid solution... try a different solver')
            return None
