# MAKE GRAPHS OF THESE COMPARISONS
# SHOW THAT CRANK NICHOLSON CONVERGES FASTER

from myBackwardEuler1Ddiffusion import BE_mse, be_duration
from myCrankNicholson1Ddiffusion import CN_mse, cn_duration
from WK8.myForwardEuler1Ddiffusion_v2 import FE_mse, fe_duration
# CAN ACCESS EVERYTHING WITHIN SCIENTIFIC COMPUTING

print(CN_mse)
print(BE_mse)
print(FE_mse)

print(cn_duration)
print(be_duration)
print(fe_duration)

