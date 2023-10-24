import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()

nodes = 1000
sigma = np.linspace(0,1,nodes)

nodes2 = 10000
sigma2 = np.linspace(0,1/2,nodes2)

#parameters
E = 15
l0 = np.sqrt(20)
Ds_i = 0
Ds_f = 0.1
l1,l2 = 0.995,1.015

    
def solve(y_init):
    def func(sigma,y):
        
        psi = y[0]
        dpsi = y[1]
        I = y[2]
        mu = y[3]
        lamb = y[4]
        J = y[5]
        
        dpsi = dpsi
        ddpsi = 4*mu*E*E*np.sin(psi)*(dpsi**2/8/E**2 - 1)/(1+mu*np.cos(psi))
        dI = np.cos(psi)*(1+dpsi**2/8/E**2)
        dmu = np.zeros_like(psi)
        dlamb = np.zeros_like(psi)
        dJ = dpsi**2/8/E**2
        
        return np.vstack((dpsi,ddpsi,dI,dmu,dlamb,dJ))
        
    def bc(ya,yb):
        
        psi0 = ya[0]
        dpsi0 = ya[1]
        I0 = ya[2]
        mu0 = ya[3]
        lamb0 = ya[4]
        J0 = ya[5]

        psi1 = yb[0]
        dpsi1 = yb[1]
        I1 = yb[2]
        J1 = yb[5]
        
        return np.array([psi0,I0,J0,psi1,I1-lamb0*(1-D),
                         lamb0**2-1-J1-mu0*lamb0*(1-D)])
    
    
    sol = intg.solve_bvp(func,bc,sigma,y_init)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    lamb = sol.y[4]
    J = sol.y[5]
    
    return np.array([psi,dpsi,I,mu,lamb,J])


    
global D


Ds = np.linspace(Ds_i,Ds_f,100)
mus_opt = np.zeros_like(Ds)
lms_opt = np.zeros_like(Ds)
x_plus_dot = np.zeros_like(Ds)
y_init = np.zeros((6,sigma.size))
z = np.pi**2/4/E**2
D_star = 1-np.sqrt(1-z)

i=0
for D in Ds:   
    print(D)
    if D<D_star:
        psi_init = np.zeros_like(sigma)
        dpsi_init = np.zeros_like(sigma)
        
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma)
        y_init[4] = 1/(1-D)*np.ones_like(psi_init)
        y_init[5] = dpsi_init
    elif D>D_star and D<D_star+0.01:
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma) * np.pi
        
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma)
        y_init[4] = np.ones_like(psi_init)
        y_init[5] = (eps*2/(1-np.pi**2/4/E**2)*np.pi)**2 \
                    *(np.pi*sigma/2+np.sin(2*np.pi*sigma)/4) #psi_init/8/E**2
        
    sol = solve(y_init)
    
    mus_opt[i] = np.mean(sol[3])
    lms_opt[i] = np.mean(sol[4])
    
    dpsi_opt = sol[1]
    p0 = dpsi_opt[0]
    f0 = 1 + p0**2/24/l0**2
    x_plus_dot[i] = E*f0/lms_opt[i]/l0-lms_opt[i]*l0*p0*np.cos(p0/2/l0)/2
    
    y_init = sol
    
    i += 1


plt.plot(Ds,x_plus_dot,'.-')
plt.ylim(-1,3.5)

f = interpolate.UnivariateSpline(Ds, x_plus_dot, s=0)
yToFind = 0
yreduced = np.array(x_plus_dot) - yToFind
freduced = interpolate.UnivariateSpline(Ds, yreduced, s=0)
D_trg = freduced.roots()[0]


plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=3)
plt.xlabel("D")
plt.ylabel(r"$\dot{x}_+$")
# plt.savefig("x_plus_dot.png",dpi=500)
plt.show()


            
def solve2(y_init):
           

    
    def func2(sigma2,y):
        
        psi = y[0]
        dpsi = y[1]
        I = y[2]
        mu = y[3]
        s_star = y[4]
        lamb = y[5]
        J = y[6]
        
        
        dpsi = dpsi
        ddpsi = 4*mu*E*E*np.sin(psi)*(dpsi**2/8/E**2/(1-2*s_star)**2 - 1)\
              *(1-2*s_star)**2/(1+mu*np.cos(psi))
        dI = np.cos(psi)*(1+dpsi**2/8/E**2/(1-2*s_star)**2)*(1-2*s_star)
        dmu = np.zeros_like(psi)
        ds_star = np.zeros_like(psi)
        dlamb = np.zeros_like(psi)
        dJ = dpsi**2/8/E**2/(1-2*s_star)
        
        return np.vstack((dpsi,ddpsi,dI,dmu,ds_star,dlamb,dJ))
        
    def bc2(ya,yb):
        psi0 = ya[0]
        dpsi0 = ya[1]
        I0 = ya[2]
        mu0 = ya[3]
        s_star0 = ya[4]
        lamb0 = ya[5]
        J0 = ya[6]

        dpsi1 = yb[1]
        I1 = yb[2]
        J1 = yb[6]
        
        roots = np.roots([l0**2 * lamb0/16/E**3, 1/24/lamb0/E**2, -l0**2 *lamb0/2/E,
                          1/lamb0])
        psi_max = roots[-1]
        dpsi_max = (-l0**2*psi_max**3/16/E**3+1/24/lamb0**2/E**2*psi_max**2+
                    l0**2/2/E*psi_max+1/lamb0**2)/(3*lamb0*l0**2/16/E**3*psi_max**2+
                    psi_max/12/lamb0/E**2-l0**2*lamb0/2/E)
        
        
        return np.array([psi0-psi_max*s_star0, dpsi0 - (1-s_star0)*psi_max, I0,J0, dpsi1,
        I1 - lamb0*(1-D)/2 + (1+psi_max**2/8/E**2)*np.sin(psi_max*s_star0)/psi_max,
        
        2*s_star0*(psi_max*dpsi_max/4/lamb0/E**2 - psi_max**2/8/lamb0**2/E**2)+
        (1-1/lamb0**2)-2*J1 + 2*mu0*(np.cos(psi_max*s_star0)/lamb0/psi_max*s_star0 \
        *dpsi_max*(1+psi_max**2/8/E**2)-np.sin(psi_max*s_star0)/lamb0*psi_max**2 \
        *dpsi_max*(1-psi_max**2/8/E**2)-1/2/lamb0*(1-D))])

    sol = intg.solve_bvp(func2,bc2,sigma2,y_init,tol=1e-8,max_nodes=10000)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    s_star = sol.y[4]
    lamb = sol.y[5]
    J = sol.y[6]
    
    
    return [psi,dpsi,I,mu,s_star,lamb,J]


print("-------------above D_trg------------")



Ds2 = np.linspace(D_trg,Ds_f,20)
mus_opt2 = np.zeros_like(Ds2)
lms_opt2 = np.zeros_like(Ds2)
s_opt2 = np.zeros_like(Ds2)
y_init = np.zeros((7,sigma2.size))

i=0
for D in Ds2:   
    print(D)

    if i < 1:
        eps = np.sqrt(1 - 1.005*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma2)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma2) * np.pi
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma2)
        y_init[4] = np.zeros_like(sigma2)
        y_init[5] = 1.01*np.ones_like(sigma2)
        y_init[6] = (eps*2/(1-np.pi**2/4/E**2)*np.pi)**2 \
                    *(np.pi*sigma2/2+np.sin(2*np.pi*sigma2)/4)/8/E**2
        

    sol  = solve2(y_init)
    lms_opt2[i] = np.mean(sol[5])
    mus_opt2[i] = np.mean(sol[3])
    s_opt2[i] = np.mean(sol[4])
    y_init = sol
    i += 1


    
    
plt.plot(Ds,lms_opt,'o-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,lms_opt2,'o-',label=r'above $D_{\Delta}$')
plt.axvline(x=D_star,linestyle='--',c='black',linewidth=2,label=r'$D^*$')
plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=2,label=r'$D_{\Delta}$')
plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.ylim(l1,l2)
plt.legend()
plt.title("Using previous initial condition")
# plt.savefig("lambda_pinit.png",dpi=500)
plt.show()


plt.plot(Ds,mus_opt,'o-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,mus_opt2,'o-',label=r'above $D_{\Delta}$')
plt.axvline(x=D_star,linestyle='--',c='black',linewidth=2,label=r'$D^*$')
plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=2,label=r'$D_{\Delta}$')
plt.xlabel("D")
plt.ylabel(r"$\mu$")
plt.ylim(0.01,0.02)
plt.legend()
plt.title("Using previous initial condition")
# plt.savefig("mu_pinit.png",dpi=500)
plt.show()


# plt.plot(Ds2,s_opt2,'o-')
# plt.show()


print("Execution time:",datetime.now() - startTime)