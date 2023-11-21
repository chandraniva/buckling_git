import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()


#parameters
E = 15
Ds_i = 0
Ds_f = 0.5
l0s = list(np.sqrt(np.arange(7,22,1)))+list(np.sqrt(np.array([50,100,150])))

Dis=[]
Dtrg=[]
l02=[]

for l0 in l0s:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+".npy")
    Dtrg.append(Ds2[0])
    if Di<Ds_f:
        l02.append(l0)
        Dis.append(Di)


x2 = np.arange(7,150,1).astype(np.float32)

l0s = np.array(l0s)

plt.plot(l0s**2,Dtrg,'o')   
plt.plot(x2,13.2*x2**-1.8,label='power law fit')
plt.xlabel(r"$l_0^2$")
plt.ylabel(r"$D_{\Delta}$")
plt.legend()
plt.loglog()
plt.title(r"$\Xi$="+str(E))
plt.savefig("Dtrg_E="+str(E)+".png")
plt.show()    
    
l02 = np.array(l02)

plt.plot(l02**2,Dis,'.-')   
plt.xlabel(r"$l_0^2$")
plt.ylabel(r"$D_{\rm intersection}$")
# plt.savefig("lms_E="+str(E)+".png",dpi=500)
plt.show()   


#parameters
l0 = np.sqrt(10)
Ds_i = 0
Ds_f = 0.5
Es = list(np.arange(10,20,1))+[30,50,100]

Dis2=[]
Dtrg2=[]
E2=[]

for E in Es:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
    Dtrg2.append(Ds2[0])
    if Di<Ds_f:
        E2.append(E)
        Dis2.append(Di)


x3 = np.arange(10,100,1).astype(np.float32)

plt.plot(Es,Dtrg2,'o')   
plt.plot(x3,0.00168*x3**1.785,label='power law fit')
plt.xlabel(r"$\Xi$")
plt.ylabel(r"$D_{\Delta}$")
plt.legend()
plt.loglog()
plt.title(r"$\l_0^2$="+str(round(l0**2)))
plt.savefig("Dtrg_l0^2="+str(10)+".png")
plt.show()    
    
E2 = np.array(E2)

plt.plot(E2,Dis2,'.-')   
plt.xlabel(r"$l_0^2$")
plt.ylabel(r"$D_{\rm intersection}$")
# plt.savefig("lms_E="+str(E)+".png",dpi=500)
plt.show()   


print("Execution time:",datetime.now() - startTime)