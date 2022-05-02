# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# %%
import numpy as np
import pylab as pl  # LIKE MATPLOTLIB
from math import pi
import scipy
from scipy import linalg
import time

# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=1.0         # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 20     # number of gridpoints in space
mt = 500   # number of gridpoints in time

# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)
# MAYBE MAKE IT INTO AN ERROR
if lmbda > 0.5:
    print('Euler stability criterion not met, solutions will be incorrect!')

# Set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jp1 = np.zeros(x.size)      # u at next time step

# INITIAL IMPLEMENTATION:
# Set initial condition
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])
    
# %%
# BACKWARD EULER
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])
be_start = time.time()
u_j_BE = u_j
lambda_array = np.array([lmbda for i in range(mx - 1)])
A_BE_1 = np.diag(1 + 2* lambda_array)
A_BE_2 = np.diag(-1 * lambda_array[:-1], k=1)
A_BE_3 = np.diag(-1 * lambda_array[:-1], k=-1)
A_BE = A_BE_1 + A_BE_2 + A_BE_3
# MATRIX be MAY CHANGE EACH TIME STEP
# A_BE * u_j+1 = u_j
# u_j+1 = inv(A_BE) * u_j
inv_A_BE = np.linalg.inv(A_BE)
for j in range(0, mt):
    u_j_BE[1:-1] = np.matmul(inv_A_BE, u_j_BE[1:-1])
    u_j_BE[[0, -1]] = 0
be_finish = time.time()
be_duration = be_finish - be_start

# Plot the final result and exact solution
pl.plot(x,u_j_BE,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

BE_mse = 0
for i in range(len(u_j_BE)):
    BE_mse += (u_j_BE[i] - u_exact(xx, T)[i])**2
BE_mse /= len(u_j_BE)

# %%
# CRANK NICHOLSON
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])

cn_start = time.time()
u_j_CN = u_j
lambda_array = np.array([lmbda for i in range(mx - 1)])
# A_CN = tridiag(-lmbda/2, 1+lambda, -lambda/2)
A_CN_1 = np.diag(1 + lambda_array)
A_CN_2 = np.diag(-1/2 * lambda_array[:-1], k=1)
A_CN_3 = np.diag(-1/2 * lambda_array[:-1], k=-1)
A_CN = A_CN_1 + A_CN_2 + A_CN_3
# B_CN = tridiag(lamdba/2, 1-lamdba, lamdba/2)
B_CN_1 = np.diag(1 - lambda_array)
B_CN_2 = np.diag(1/2 * lambda_array[:-1], k=1)
B_CN_3 = np.diag(1/2 * lambda_array[:-1], k=-1)  # CAN JUST USE MATRIX TRANSPOSE OF B_CN_2
B_CN = B_CN_1 + B_CN_2 + B_CN_3
# A_CN * u_j+1 = B_CN * u_j
# u_j+1 = inv(A_CN) * B_CN * u_j
for j in range(0, mt):
    # test = u_j_CN[1:-1]
    # u_j_CN[1:-1] = inv_A_CN @ B_CN @ test
    b = B_CN @ u_j_CN[1:-1]
    u_j1_CN = scipy.linalg.solve(A_CN, b)
    u_j_CN[1: -1] = u_j1_CN  
cn_finish = time.time()
cn_duration = cn_finish - cn_start

# %%
# WHY CANT YOU USE MATRIX INVERSION?
# mat inv in python fucks up when the determinant of the matrix is too close to 0
# OR when eigenvalues of the matrix are too close to each other
A_CN_det = np.linalg.det(A_CN)
w, v = np.linalg.eig(A_CN)
print('determinant of matrix is A_CN is', A_CN_det)
print('eigenvalues are as follows:')
print(w)
# as shown the issue with using matrix inversion in this case is that the eigenvalues are all very close together
# therefore matrix inversion operation is faulty, therefore solutions are wrong
# otherwise the algorithm is working fine, and in this specific case linear solvers should be used
# IT IS ALWAYS ADVISED TO USE LINEAR SOLVERS

# %%
#AB_CN = np.matmul(inv_A_CN, B_CN)
for j in range(0, mt):
    # test = u_j_CN[1:-1]
    # u_j_CN[1:-1] = inv_A_CN @ B_CN @ test
    b = B_CN @ u_j_CN[1:-1]
    u_j1_CN = scipy.linalg.solve(A_CN, b)
    u_j_CN[1: -1] = u_j1_CN  

# Plot the final result and exact solution
pl.plot(x,u_j_CN,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

CN_mse = 0
for i in range(len(u_j_CN)):
    CN_mse += (u_j_CN[i] - u_exact(xx,T)[i])**2
CN_mse /= len(u_j_CN)

# %%
# FORWARD EULER
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])

method = 'Matrix'
match method:
    case 'singular':
        # Solve the PDE: loop over all time points
        for j in range(0, mt):
            # Forward Euler timestep at inner mesh points
            # PDE discretised at position x[i], time t[j]
            for i in range(1, mx):
                u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])  # find solution forward 1 step in time
                
            # Boundary conditions
            u_jp1[0] = 0; u_jp1[mx] = 0
                
            # Save u_j at time t[j+1]
            u_j[:] = u_jp1[:]

    case 'Matrix':
        # MATRIX IMPLEMENTATION
        fe_start = time.time()
        lambda_array = np.array([lmbda for i in range(mx - 1)])
        A_FE_1 = np.diag(1 - 2* lambda_array)
        A_FE_2 = np.diag(lambda_array[:-1], k=1)
        A_FE_3 = np.diag(lambda_array[:-1], k=-1)
        A_FE = A_FE_1 + A_FE_2 + A_FE_3

        for j in range(0, mt):
            u_j[1:-1] = np.matmul(A_FE, u_j[1:-1])
            u_j[[0,-1]] = 0
        fe_finish = time.time()
fe_duration = fe_finish - fe_start

# Plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

FW_mse = 0
for i in range(len(u_j)):
    FW_mse += (u_j[i] - u_exact(xx,T)[i])**2
FW_mse /= len(u_j)
# %%
# EVALUATION
mse = {'FE' : (FW_mse, fe_duration), 'BE' : (BE_mse, be_duration), 'CN' : (CN_mse, cn_duration)}
min = np.inf
min_mse = None
for i in mse:
    if mse[i] < min:
        min_mse = i
        min = mse[i]
