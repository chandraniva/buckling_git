import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()


#parameters
E = 15
Ds_i = 0
Ds_f = 0.6
l0s = np.sqrt(np.arange(7,18,1))

tol = 1e-3*3

"""-----------lambda------------"""

for l0 in l0s:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,lms2,'.-',label=r'$l_0^2=$'+str(round(l0**2)),zorder=0)
    if Di<Ds_f:
        plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)


    
    
plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.xlim(0,Ds_f)
plt.legend(loc =2,prop={'size':7})
plt.savefig("lms_E="+str(E)+"_tol="+str(tol)+".png",dpi=500)
# plt.savefig("bif_lms_E="+str(E)+".png",dpi=500)
plt.show()    


"""-----------mus-------------"""

for l0 in l0s:    
    Ds2,mus2 = np.load("data/mus_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,mus2,'.-',label=r'$l_0^2=$'+str(round(l0**2)),zorder=0)
    if Di<Ds_f:
        plt.scatter(Di,mus2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)
        
        

    

plt.xlabel("D")
plt.ylabel(r"$\mu$")
plt.xlim(0,Ds_f)
plt.legend(loc=2,prop={'size':7})
plt.savefig("mus_E="+str(E)+"_tol="+str(tol)+".png",dpi=500)
# plt.savefig("bif_mus_E="+str(E)+".png",dpi=500)
plt.show()


"""------------amplitude-----------------"""

Ds1,amp1 = np.load("data/amp1_E="+str(E)+".npy")

i=0
for l0 in l0s:
    
    Ds1,amp1 = np.load("data/amp1_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Ds2,amp2 = np.load("data/amp_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    i_trg = np.where(abs(Ds1-Ds2[0])==min(abs(Ds1 - Ds2[0])))[0][0]
    
    plt.plot(Ds1[:i_trg],amp1[:i_trg],'.-',color='C'+str(i),zorder=0)
    plt.plot(Ds2,amp2,'.-',color='C'+str(i),label=r'$l_0^2=$'+str(round(l0**2)),zorder=0)
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    if Di<Ds_f:
        plt.scatter(Di,amp2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)
    i+=1
    
    

    
plt.xlabel("D")
plt.ylabel("Buckling amplitude")
plt.xlim(-0.1,Ds_f)
# plt.xlim(0.20,0.25)
# plt.ylim(2,2.5)
plt.legend(loc=2,prop={'size':7})
plt.savefig("amps_E="+str(E)+"_tol="+str(tol)+".png",dpi=500)
# plt.savefig("bif_amps_E="+str(E)+".png",dpi=500)
plt.show()

print("Execution time:",datetime.now() - startTime)