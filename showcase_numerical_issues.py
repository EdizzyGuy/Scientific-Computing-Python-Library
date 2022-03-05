import numpy as np
import Euler
import matplotlib.pyplot as plt

# What goes wrong with the numerical solutions if you run them over a large range of t? (This is clearer if you plot x
# against x_dot rather than x against t and if you use time steps that are not very small.)
# WRITE ABOUT WHAT GOES WRONG MORE ANALYTICALLY / MATHEMATICALLY


def dX_dt(X):
    X_dot = np.array([X[1], -X[0]])
    return X_dot


initial_condition = np.array([0, 1])
time_step = 1
solve_for = np.linspace(0, 200*3, 1001)

result_euler = Euler.solve_ode(dX_dt, initial_condition, solve_for, deltat_max=time_step, method='Euler')
result_rk4 = Euler.solve_ode(dX_dt, initial_condition, solve_for, deltat_max=time_step, method='RK4')
true_results = np.array([np.sin(solve_for), np.cos(solve_for)]).transpose()

fig = plt.figure(figsize=(15, 6), dpi=100)
fig.suptitle(r"$\bf{Graph\ of\ x\ against\ \dot{x}}$"
             "\nShowcasing numerical issues with solve_ode function and large time steps, for large ranges of time.",
             fontsize=12)

ax1 = plt.subplot(2, 6, (1, 2))
plt.plot(result_euler[:200, 0], result_euler[:200, 1], color='blue', linewidth=2.5, linestyle='-')
ax1.set_xlabel(r'$\bf{x}$')
ax1.set_ylabel(r'$\bf{\dot{x}}$', rotation=0)
ax1.set_title(r"Euler's method", fontsize=8)
plt.grid()

ax2 = plt.subplot(2, 6, (3, 4))
plt.plot(result_rk4[:, 0], result_rk4[:, 1], color='blue', linewidth=2.5, linestyle='-')
ax2.set_xlabel(r'$\bf{x}$')
ax2.set_title(r"Classic Runge-Kutta method", fontsize=8)
plt.grid()

ax3 = plt.subplot(2, 6, (5, 6))
plt.plot(true_results[:, 0], true_results[:, 1], color='red', linewidth=2.5, linestyle='-')
ax3.set_title(r"True solution", fontsize=8)
ax3.set_xlabel(r'$\bf{x}$')
plt.grid()

ax4 = plt.subplot(2, 6, (7, 9))
plt.plot(solve_for[:20], true_results[:20, 0], color='red', linewidth=2.5, linestyle='-', label='True solution')
plt.plot(solve_for[:20], result_euler[:20, 0], color='blue', linewidth=2.5, linestyle='-', label='Euler approximation')
plt.legend(loc='upper left', frameon=True)
ax4.set_title(r"$\bf{Graph\ of\ x\ against\ time$")
ax4.set_xlabel(r'$\bf{t}$')
ax4.set_ylabel(r'$\bf{x}$', rotation=0)

ax5 = plt.subplot(2, 6, (10, 12))
plt.plot(solve_for[:-200], true_results[:-200, 0], color='red', linewidth=2.5, linestyle='-', label='True solution')
plt.plot(solve_for[:-200], result_rk4[:-200, 0], color='blue', linewidth=2.5, linestyle='-', label='Runge-Kutta\napproximation')
plt.legend(loc='upper left', frameon=True)
ax5.set_title(r"$\bf{Graph\ of\ x\ against\ time$")
ax5.set_xlabel(r'$\bf{t}$')

plt.subplots_adjust(hspace=0.42)
plt.text(0.511, 0.45, r'Approximations of $\bf{x}$ and true solution in time', transform=fig.transFigure,
         horizontalalignment='center', weight='bold', fontsize=12)
fig.savefig('numerical_issues.png')
# y_lim = (ax2.get_ylim()[0], ax1.get_ylim()[1])
# ax1.set_ylim(y_lim)
# ax2.set_ylim(y_lim)
