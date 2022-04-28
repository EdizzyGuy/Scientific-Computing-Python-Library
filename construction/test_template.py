

init_cond = np.array([1,1])
#np.random.normal(size=2) * 9
period_guess = np.random.uniform(low=5, high=25, size=1)
init_hopf = np.concatenate((init_cond, period_guess))

init_cond = np.array([1,1,1])
init_hopf_ext = np.concatenate((init_cond, period_guess))

beta, sigma = 2,-1
atol, rtol = 1e-2, 1e-2
# from analytical solution
anal_period = 2* PI
anal_radius = np.sqrt(beta)

roots = find_limit_cycles(hopf_bifurcation, init_hopf, args=(beta, sigma))
#find_limit_cycles(hopf_bifurcation, init_guess, args=(beta, sigma))
numer_period = roots[-1]
numer_radius = np.linalg.norm(roots[:-1])

# check if period is a multiple of the analytical sol
period_multiplicity = numer_period / anal_period
period_multiple = np.isclose(period_multiplicity % 1, 1, atol=atol) or np.isclose(period_multiplicity % 1, 0, atol=atol)
print('HOPF SOL')
print(f'Analytical and numerical radius are numerically equivalent : {np.isclose(anal_radius, numer_radius, rtol=rtol)}')
print(f'Numerical solution to radius is integer multiple of analytical solution : {period_multiple}')
print()

roots = find_limit_cycles(hopf_extended, init_hopf_ext, args=(beta, sigma))
#find_limit_cycles(hopf_bifurcation, init_guess, args=(beta, sigma))
numer_period = roots[-1]
numer_radius = np.linalg.norm(roots[:-1])

# check if period is a multiple of the analytical sol
period_multiplicity = numer_period / anal_period
period_multiple = np.isclose(period_multiplicity % 1, 1, atol=atol) or np.isclose(period_multiplicity % 1, 0, atol=atol)
print('HOPF EXT SOL')
print(f'Analytical and numerical radius are numerically equivalent : {np.isclose(anal_radius, numer_radius, rtol=rtol)}')
print(f'Numerical solution to radius is integer multiple of analytical solution : {period_multiple}')
print()