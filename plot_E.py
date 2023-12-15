import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()


#parameters
Ds_i = 0
Ds_f = 0.8
Es = np.concatenate( (np.arange(16,32,2), np.array([27])), axis = None)
l0 = np.sqrt(20)

tol = 1e-3*5


"""-----------lambda------------"""

i=0
for E in Es:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,lms2,'.-',c='C'+str(i),label=r'$\Xi=$'+str(E),zorder=0)
    # plt.scatter(Ds2[0],lms2[0],s=50,color='C'+str(i),zorder=1)
    if Di<Ds_f and Di>Ds2[0]:
        plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)
    i+=1

plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.xlim(-0.1,Ds_f)
plt.xlim(-0.1,0.6)
plt.ylim(0.75,1.50)
plt.legend(loc =2,prop={'size':7})
plt.savefig("lms_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".png",dpi=500)
plt.show()   
    

"""-----------mus-------------"""


for E in Es:    
    Ds2,mus2 = np.load("data/mus_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,mus2,'.-',label=r'$\Xi=$'+str(E),zorder=0)
    plt.scatter(Ds2[0],mus2[0],s=50,c='yellow',zorder=1)
    if Di<Ds_f and Di>Ds2[0]:
        plt.scatter(Di,mus2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)


plt.xlabel("D")
plt.ylabel(r"$\mu$")
plt.xlim(-0.1,Ds_f)
plt.xlim(-0.1,0.6)
plt.ylim(-0.05,0.4)
plt.legend(loc=2,prop={'size':7})
plt.savefig("mus_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".png",dpi=500)
plt.show()   


"""------------amplitude-----------------"""


i=0
for E in Es:
    
    Ds1,amp1 = np.load("data/amp1_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Ds2,amp2 = np.load("data/amp_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    i_trg = np.where(abs(Ds1-Ds2[0])==min(abs(Ds1 - Ds2[0])))[0][0]
    
    plt.plot(Ds1[:i_trg],amp1[:i_trg],'.-',color='C'+str(i),zorder=0)
    plt.plot(Ds2,amp2,'.-',color='C'+str(i),label=r'$\Xi=$'+str(E),zorder=0)
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    plt.scatter(Ds2[0],amp2[0],s=50,c='yellow',zorder=1)
    if Di<Ds_f and Di>Ds2[0]:
        plt.scatter(Di,amp2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)
    elif Di<Ds2[0]:
        plt.scatter(Di,amp1[np.where(Ds1==Di)[0]],s=50,c='cyan',zorder=1)
    i+=1

plt.xlabel("D")
plt.ylabel("Buckling amplitude")
plt.xlim(-0.1,Ds_f)
plt.legend(loc=2,prop={'size':7})
plt.savefig("amps_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".png",dpi=500)
plt.show()

print("Execution time:",datetime.now() - startTime)