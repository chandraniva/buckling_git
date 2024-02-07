import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()



r = 1.16
D_bif = 0.2115#1-np.sin(r)/r
xi = 1/100

params = [[10,8.14],[15,12.59],[25,21.42],
          [100,85.6]]

# for p in params:
#     E, l0_sq = p
#     Ds2, s2 = np.load("data/s_E="+str(E)+"_l0^2="+str(l0_sq)+".npy")
    
    
Ds2, s = np.load("data/s_E="+str(100)+"_l0^2="+str(85.8)+".npy")
delta = np.linspace(0,-1000,100)
s1 = ((-12*delta*(1/np.sin(r))+np.tan(r))/8/r)**0.5

plt.plot(D_bif-Ds2,-s+0.5,'o-')
plt.plot(-delta*xi**2,s1*xi,label='theory')
# plt.plot(x,1.2*np.sqrt(x),label='1.2*delta^1/2')
plt.xlim(0.001,0.1)
plt.ylim(0.01,1)
plt.xlabel("D_bif-D")
plt.ylabel("1/2-s*")
plt.legend()
plt.loglog()
plt.savefig("s_scaling.png",dpi=500)

