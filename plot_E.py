import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()


#parameters
Ds_i = 0
Ds_f = 0.5
Es = np.arange(10,20,1)
l0 = np.sqrt(10)

for E in Es:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")

    plt.plot(Ds2,lms2,'.-',label=r'$\Xi=$'+str(E),zorder=0)
    if Di<Ds_f:
        plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)


plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.xlim(0.0,Ds_f)
plt.ylim(0.85,1.20)
plt.legend(loc =2,prop={'size':7})
plt.savefig("lms_l0^2="+str(round(l0**2*10)/10)+".png",dpi=500)
plt.show()    
    

for E in Es:    
    Ds2,mus2 = np.load("data/mus_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
    
    plt.plot(Ds2,mus2,'.-',label=r'$\Xi=$'+str(E),zorder=0)
    if Di<Ds_f:
        plt.scatter(Di,mus2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)


plt.xlabel("D")
plt.ylabel(r"$\mu$")
plt.xlim(0,Ds_f)
plt.ylim(-0.01,1)
plt.legend(loc=2,prop={'size':7})
plt.savefig("mus_l0^2="+str(round(l0**2*10)/10)+".png",dpi=500)
plt.show()


print("Execution time:",datetime.now() - startTime)