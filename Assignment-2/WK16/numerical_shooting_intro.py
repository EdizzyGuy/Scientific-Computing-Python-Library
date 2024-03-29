import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

'''
Q1
i   simulate predator prey equations

ii  what is behaviour in long time limit for
        b > 0.26, b < 0.26, b = 0.26

iii isolate a peridoic orbit
        what is its period and starting conditions

THESE ARE TESTS FOR NUMERICAL METHODS
    
Q2
    determine an appropiate phase condition
    
Q3
    Construct the shooting root-finding problem for the predator-prey example; 
    check that it can find the periodic orbit found in 1. 
    
Q4
    Generalise your code so that you can use arbitrary differential equations of arbitrary dimension 
    (assume they are always in first-order form).
    - How should a user pass the differential equations to your code?
    - How should a user pass the phase-condition to your code?
    - What options might a user want to have access to?
'''

# predator prey equations:
def dXdt(t, X, a, b, d):
    x = X[0]
    y = X[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b *y *(1 - y/x)

    dXdt = np.array([dxdt, dydt])
    return dXdt


a, d = 1, 0.1

# what behavior do you see for b > 0.26 in the long run
t = np.linspace(0, 100, 1000)
t_span = (t[0], t[-1])
b_array = np.linspace(0.1, 0.5, 4)
rand_init_conditions = [np.random.rand(2) for _ in range(5)]

result = solve_ivp(dXdt, t_span, rand_init_conditions[0], args=(a, b_array[0], d))
plt.plot(result[:, 0], result.y[:, 1])
ax1.set_xlabel('$')
ax1.set_ylabel(r'$\bf{\dot{x}}$', rotation=0)
ax1.set_title(r"Euler's method", fontsize=8)


'''
fig = plt.figure(figsize=(8,6), dpi=100)
for _, b in enumerate(b_array):
    ax = plt.subplot(2,2,_+1) 
    for init_cond in rand_init_conditions:
       result = solve_ivp(dXdt, t_span, init_cond) 
       plt.plot(result[:, 0], result[:, 1], color='blue')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    ax.set_title(f'b = {b:.2}')
fig.suptitle(r'Phase portraits of system for changing parameter, b') '''

