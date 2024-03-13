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
Es =  (np.arange(8,25,1))#, np.array([])), axis = None)
l0 = np.sqrt(10)

tol = 1e-3*3


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
# plt.savefig("lms_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".png",dpi=500)
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
# plt.savefig("mus_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".png",dpi=500)
plt.show()   


"""------------amplitude-----------------"""

plt.figure(figsize=(7,5))
colors = plt.cm.viridis(np.linspace(0, 1, 17))

i=0
for E in Es:
    
    Ds1,amp1 = np.load("data/amp1_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    Ds2,amp2 = np.load("data/amp_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    i_trg = np.where(abs(Ds1-Ds2[0])==min(abs(Ds1 - Ds2[0])))[0][0]
    
    plt.plot(Ds1[:i_trg],amp1[:i_trg],'.-',color=colors[i],zorder=0)
    
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".npy")
    
    i_ints = np.where(abs(Ds2-Di) == min(abs(Ds2-Di)))[0][0]
    
    plt.scatter(Ds2[0],amp2[0],s=50,c='red',zorder=1)
    
    # if Di<Ds_f and Di>Ds2[0]:
    #     plt.scatter(Di,amp2[np.where(Ds2==Di)[0]],s=50,c='cyan',zorder=2)
    # elif Di<Ds2[0]:
    #     plt.scatter(Di,amp1[np.where(Ds1==Di)[0]],s=50,c='cyan',zorder=2)
        
    plt.plot(Ds2[:i_ints],amp2[:i_ints],'.-',color=colors[i],zorder=0)
    i+=1
    
colorbar_min = 8/l0**2
colorbar_max = 25/l0**2
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=colorbar_min, vmax=colorbar_max))
sm.set_array([])  # You need to set a dummy array for the scalar mappable
cbar = plt.colorbar(sm)
cbar.ax.tick_params(labelsize=17) 

plt.xlabel("D",fontsize=18)
plt.ylabel("Buckling amplitude",fontsize=15)
plt.xlim(0,0.6)
# plt.ylim(-0,8)
plt.legend(loc=2,fontsize=10,ncol=2)
# plt.title(r"$l_0^2 = 10$",fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.savefig("amps_l0^2="+str(round(l0**2*10)/10)+"_tol="+str(tol)+".png",dpi=500)
plt.show()

print("Execution time:",datetime.now() - startTime)