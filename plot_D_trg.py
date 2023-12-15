import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()



"""  --------- with l0 -------------"""


tol=1e-3*3

# parameters
E = 30
Ds_i = 0
Ds_f = 1
l0s = list(np.sqrt(np.arange(10,40,2)))+list(np.sqrt(np.array([42,46,54,60,
                                                               70,80,100,120,
                                                               200,300,
                                                               500,800])))


Dis=[]
Dtrg=[]
l02=[]

for l0 in l0s:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
            "_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
            "_tol="+str(tol)+".npy")
    
    Dtrg.append(Ds2[0])
    if Di<Ds_f:
        l02.append(l0)
        Dis.append(Di)


"""---------estimate--------"""

l0s2 = np.sqrt(np.linspace(10,800,50))
pi = np.pi
D_trgs = np.zeros_like(l0s2)

i = 0
for l0 in l0s2:
    xi = pi/E
    r = E/l0**2
    z = pi/(4*E**2)
    eps0 = 2*np.sqrt((4-xi**2)/(32-14*xi**2+xi**4))
    psi0 = 2/np.sqrt(1-z)
    D_star = 1 - np.sqrt(1-z)
    
    D_trgs[i] = r**2/pi**2*(1-z)**2*(1-7/4*z+z**2/2)+D_star
    
    
    i+=1

"""***********************"""

l02 = np.array(l02)
l0s = np.array(l0s)

x2 = np.arange(10,120,1).astype(np.float32)


plt.plot(l0s**2,Dtrg,'o',label=r'$D_{\Delta}$')  
plt.plot(l0s2**2,D_trgs,'red',label='theory: estimate') 
# plt.plot(l02[:-9]**2,Dis[:-9],'.-',label=r'$D_{{intersection}}$')  
# plt.plot(x2,55*x2**-1.85,'r',label='power law fit')
plt.xlabel(r"$l_0^2$")
plt.ylabel(r"$D_{\Delta}$")
plt.legend()
plt.loglog()
plt.title(r"$\Xi$="+str(E))
plt.savefig("Dtrg_D_int_E="+str(E)+".png",dpi=500)
plt.show()    
    


"""  --------- with E -------------"""


#parameters
l0 = np.sqrt(50)
Ds_i = 0
Ds_f = 1.0
Es = list(np.arange(10,155,5))#+[120,140,150]

Dis2=[]
Dtrg2=[]
E2=[]

for E in Es:
    Ds2,lms2 = np.load("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
            "_tol="+str(tol)+".npy")
    Di = np.load("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
            "_tol="+str(tol)+".npy")
    Dtrg2.append(Ds2[0])
    if Di<Ds_f:
        E2.append(E)
        Dis2.append(Di)



"""---------estimate--------"""

Es2 = np.linspace(10,155,100)
pi = np.pi
D_trgs = np.zeros_like(Es2)

i = 0
for E in Es2:
    xi = pi/E
    r = E/l0**2
    z = pi/(4*E**2)
    eps0 = 2*np.sqrt((4-xi**2)/(32-14*xi**2+xi**4))
    psi0 = 2/np.sqrt(1-z)
    D_star = 1 - np.sqrt(1-z)
    
    roots = np.roots([-eps0**2*psi0**2*pi**2/4/E**2/(1-D_star)+
                      eps0**4*psi0**2*pi**2/4/E**2, 0, 
                      2/(1-D_star)-2*eps0**2-eps0**2*psi0**2*pi**2/8/E**2,
                      -eps0*psi0*z/3*(1-D_star)**2*r,1,
                      -2*(1-D_star)**2*r/eps0/psi0/pi])
    # print(roots)
    D_trgs[i] = r**2/pi**2*(1-z)**2*(1-7/4*z+z**2/2)+D_star#(np.real(roots[4]))**2 + D_star
    
    
    i+=1
    
"""***********************"""

alpha = 1.9
fit_str = r'power law: $\alpha=$'+str(alpha)

x3 = np.arange(10,155,1).astype(np.float32)

plt.plot(Es,Dtrg2,'o',label=r'$D_{\Delta}$')  
# plt.plot(E2[1:],Dis2[1:],'o-',label=r'$D_{{intersection}}$')  
plt.plot(Es2,D_trgs,'red',label='theory: estimate') 
# plt.plot(x3,0.000057*x3**1.9,color='red',label=fit_str)
plt.xlabel(r"$\Xi$")
plt.ylabel(r"$D_{\Delta}$")
plt.legend()
plt.loglog()
plt.title(r"$\l_0^2$="+str(round(l0**2)))
plt.savefig("Dtrg_D_int_l0^2="+str(10)+".png",dpi=500)
plt.show()    
    



print("Execution time:",datetime.now() - startTime)