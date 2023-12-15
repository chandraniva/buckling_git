import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()


Ds_i = 0
Ds_f = 0.4

D_bif = 0.213

tol = 1e-3*3

params = [[10,8.14],[10,8.13],[15,12.59],[15,12.62],[25,21.42],[25,21.41],
          [100,85.7],[100,85.8]]


"""---------------------- Lambda ------------------------"""

for p in params:
    E, l0_sq = p
        
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    
    Ds2,s2 = np.load("data/s_star_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(l0_sq)+
                 "_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,lms2,'.-',label=r'$l_0^2=$'+str(l0_sq)+'; $\Xi=$'+str(E),zorder=0)
    
    # if Di<Ds_f:
    #     plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='red',zorder=1)

# plt.figure(figsize=(10,10))
# plt.plot(Ds2,s2,'.-',c='blue',label=r'$s^{*}$ for $\Xi_c$='+str(E))
# plt.axvline(x=D_bif,c='black',linestyle='--')
# plt.axhline(y=0.5,c='black',linestyle='--')
plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.xlim(0.1,0.25)
plt.ylim(0.95,1.05)
plt.legend(loc =2,prop={'size':5})
# plt.savefig("bif_lms.png",dpi=500)
plt.show()    


"""---------------------- mu ------------------------"""


for p in params:
    E, l0_sq = p
        
    Ds2,mu2 = np.load("data/mus_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(l0_sq)+
                 "_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,mu2,'.-',label=r'$l_0^2=$'+
              str(l0_sq)+'; $\Xi=$'+str(E),zorder=0)
    # if Di<Ds_f:
    #     plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='red',zorder=1)

plt.plot(Ds2,s2,'.-',c='blue',label=r'$s^{*}$ for $\Xi_c$='+str(E))
plt.axvline(x=D_bif,c='black',linestyle='--')
plt.axhline(y=0.5,c='black',linestyle='--')
plt.xlabel("D")
plt.ylabel(r"$\mu$")
plt.xlim(0,Ds_f)
plt.ylim(0,1)
plt.legend(loc=2,prop={'size':5})
# plt.savefig("bif_mus.png",dpi=500)
plt.show()



"""----------------------- Amplitude ------------------------"""

i=0
for p in params:
    E, l0_sq = p
        
    Ds1,amp1 = np.load("data/amp1_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    Ds2,amp2 = np.load("data/amp_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    
    i_trg = np.where(abs(Ds1-Ds2[0])==min(abs(Ds1 - Ds2[0])))[0][0]
    
    plt.plot(Ds1[:i_trg],amp1[:i_trg],'.-',color='C'+str(i),zorder=0)
    
    plt.plot(Ds2,amp2,'.-',color='C'+str(i),label=r'$l_0^2=$'+
             str(l0_sq)+'; $\Xi=$'+str(E),zorder=0)
    
    # Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+
    #              "_tol="+str(tol)+".npy")
    # if Di<Ds_f:
    #     plt.scatter(Di,amp2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=1)
    i+=1
    

plt.plot(Ds2,s2,'.-',c='blue',label=r'$s^{*}$ for $\Xi_c$='+str(E))
plt.axvline(x=D_bif,c='black',linestyle='--')
plt.axhline(y=0.5,c='black',linestyle='--')
plt.xlabel("D")
plt.ylabel("Buckling amplitude")
plt.xlim(-0.1,Ds_f)
plt.legend(loc=2,prop={'size':5})
# plt.savefig("bif_amps.png",dpi=500)
plt.show()


"""-------------------- psi_max -----------------------"""

flag = 0
for p in params:
    E, l0_sq = p
    

    
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    
    Ds2,s2 = np.load("data/s_star_E="+str(E)+"_l0^2="+str(l0_sq)+
                       "_tol="+str(tol)+".npy")
    psi_max = np.zeros_like(lms2)
    
    l0 = np.sqrt(l0_sq)
    
    i = 0
    for lamb in lms2:
        roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
        if roots[0] == complex() or roots[1]==complex():
            flag =1 
        
        psi_max[i] = roots[-1]
        i+=1
    
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(l0_sq)+
                 "_tol="+str(tol)+".npy")
    
    plt.plot(Ds2,psi_max,'.-',label=r'$l_0^2=$'+str(l0_sq)+'; $\Xi=$'+str(E),zorder=0)
    
    # if Di<Ds_f:
    #     plt.scatter(Di,lms2[np.where(Ds2==Di)[0]],s=50,c='red',zorder=1)

# plt.figure(figsize=(10,10))
# plt.plot(Ds2,s2,'.-',c='blue',label=r'$s^{*}$ for $\Xi_c$='+str(E))
# plt.axvline(x=D_bif,c='black',linestyle='--')
# plt.axhline(y=0.5,c='black',linestyle='--')
plt.xlabel("D")
plt.ylabel(r"$\psi_{max}$")
# plt.xlim(0.1,0.25)
# plt.ylim(0.95,1.05)
plt.legend(loc =2,prop={'size':5})
plt.savefig("bif_psi_max.png",dpi=500)
plt.show()   

print("flag=",flag)


print("Execution time:",datetime.now() - startTime)