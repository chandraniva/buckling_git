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
l0s = np.sqrt(np.arange(7,17,1))


# for l0 in l0s:
#     Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
#     Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")

#     plt.plot(Ds2,lms2,'.-',label=r'$l_0^2=$'+str(round(l0**2)),zorder=0)
#     if Di<Ds_f:
#         plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)

Ds2,lms2a = np.load("data/lms_E="+str(E)+"_l0^2=12.551.npy")
Dia = np.load("data/intsct_E="+str(E)+"_l0^2=12.551.npy")

plt.plot(Ds2,lms2a,'.-',c='green',zorder=0)#,label=r'$l_0^2=12.551$',zorder=0)
if Dia<Ds_f:
    plt.scatter(Dia,lms2a[np.where(Ds2==Dia)[0]],s=50,c='red',zorder=1)
    
Ds2,lms2b = np.load("data/lms_E="+str(E)+"_l0^2=12.554.npy")
Dib = np.load("data/intsct_E="+str(E)+"_l0^2=12.554.npy")

plt.plot(Ds2,lms2b,'.-',c='green')#, label=r'$l_0^2=12.553$',zorder=0)
if Dib<Ds_f:
    plt.scatter(Dib,lms2b[np.where(Ds2==Dib)[0]],s=50,c='red',zorder=1)
    
    
plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
# plt.xlim(0,Ds_f)
plt.legend(loc =2,prop={'size':7})
# plt.savefig("lms_E="+str(E)+".png",dpi=500)
plt.savefig("bif_lms_E="+str(E)+".png",dpi=500)
plt.show()    
    

# for l0 in l0s:    
#     Ds2,mus2 = np.load("data/mus_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
#     Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+".npy")
    
#     plt.plot(Ds2,mus2,'.-',label=r'$l_0^2=$'+str(round(l0**2)),zorder=0)
#     if Di<Ds_f:
#         plt.scatter(Di,mus2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)
        
        
Ds2,mus2a = np.load("data/mus_E="+str(E)+"_l0^2=12.55.npy")
Dia = np.load("data/intsct_E="+str(E)+"_l0^2=12.55.npy")

plt.plot(Ds2,mus2a,'.-',c='C0',zorder=0)#,label=r'$l_0^2=12.551$',zorder=0)
if Dia<Ds_f:
    plt.scatter(Dia,mus2a[np.where(Ds2==Dia)[0]],s=50,c='red',zorder=1)
    
Ds2,mus2b = np.load("data/mus_E="+str(E)+"_l0^2=12.554.npy")
Dib = np.load("data/intsct_E="+str(E)+"_l0^2=12.554.npy")

plt.plot(Ds2,mus2b,'.-',c='C0')#,label=r'$l_0^2=12.553$',zorder=0)
if Dib<Ds_f:
    plt.scatter(Dib,mus2b[np.where(Ds2==Dib)[0]],s=50,c='red',zorder=1)
    

plt.xlabel("D")
plt.ylabel(r"$\mu$")
# plt.xlim(0,Ds_f)
plt.legend(loc=2,prop={'size':7})
# plt.savefig("mus_E="+str(E)+".png",dpi=500)
plt.savefig("bif_mus_E="+str(E)+".png",dpi=500)
plt.show()


print("Execution time:",datetime.now() - startTime)