import numpy as np

def euler_step(dx_dt, initial_point, step_size):
    ''' Will implement an euler step given the derivative of a system of equations (dx_dt)
    an initial point and a step size '''
    increment = dx_dt(initial_point) * step_size
    new_point = initial_point + increment
    return new_point

def solve_ode_euler(dx_dt, initial_condition, step=0.001, time_interval=[0, 1]):
    pass