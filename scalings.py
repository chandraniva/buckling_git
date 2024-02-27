import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()

r = 1.1655
D_bif = 0.2115
E = 100
l0 = np.sqrt(85.8)
xi = 1/100
tol=1e-3*3
    
    
Ds2, s = np.load("data/s_E="+str(100)+"_l0^2="+str(85.8)+".npy")
delta = np.linspace(0,-1000,100)
s1 = ((-12*delta/np.sin(r)+np.tan(r))/8/r)**0.5

plt.plot(D_bif-Ds2,-s+0.5,'o-')
plt.plot(-delta*xi**2,s1*xi,label='theory')
# plt.plot(x,1.2*np.sqrt(x),label='1.2*delta^1/2')
plt.xlim(0.001,0.1)
plt.ylim(0.01,1)
plt.xlabel("D_bif-D")
plt.ylabel("1/2-s*")
plt.title("negative s scaling")
plt.legend()
plt.loglog()
plt.savefig("s_scaling.png",dpi=500)
plt.show()


delta2 = np.linspace(0,1000,10000)
Ds2, lms = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy")
lambda1 = (1/24*(-1-24*delta2*np.cos(r)+np.cos(2*r))/np.cos(2*r))**0.5

plt.plot(-D_bif+Ds2,lms-1,'o-')
plt.plot(delta2*xi**2,lambda1*xi,label='theory')
plt.xlabel("D-D_bif")
plt.ylabel("lambda-1")
plt.title("positive lambda scaling")
plt.loglog()
plt.savefig("lambda_scaling.png",dpi=500)
plt.show()



Ds2, amp = np.load("data/amp_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy")
amp0 = 2*(np.sin(r/2))**2 /np.sqrt(r*xi)
amp1 = -8*r*lambda1*((np.sin(r/2))**2 - r*np.sin(r))/4/r**1.5

plt.plot(-D_bif+Ds2,-amp+amp0,'o-')
plt.plot(delta2*xi**2,amp1*np.sqrt(xi),label='theory')
plt.xlabel("D-D_bif")
plt.ylabel("amp-amp0")
plt.title("positive amplitude scaling")
plt.loglog()
plt.savefig("amp_scaling.png",dpi=500)
plt.show()