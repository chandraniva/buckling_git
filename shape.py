import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()

nodes = 1000
sigma = np.linspace(0,1,nodes)

nodes2 = 1000
sigma2 = np.linspace(0,1/2,nodes2)


#parameters
E = 15
l0_sq = 12.59
l0 = np.sqrt(l0_sq)
Ds_i = 0
Ds_f = 0.3

    
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

def shape1(psi,dpsi,lamb,mu,D):
    ddpsi = 4*mu*E**2*np.sin(psi)/(1+mu*np.cos(psi))*(dpsi**2/8/E**2 - 1)
    phi = dpsi/2/E
    f = 1+dpsi**2/24/E**2
    g = ddpsi/12/E**2 
    dx = (f*np.cos(psi) - g*np.sin(psi))/lamb*E/l0
    dy = (f*np.sin(psi) - g*np.cos(psi))/lamb*E/l0
    x =  np.cumsum(dx)/nodes 
    y =  np.cumsum(dy)/nodes
    xp = x - l0*lamb/2*np.sin(psi)*np.cos(phi)
    xm = x + l0*lamb/2*np.sin(psi)*np.cos(phi)
    yp = y + l0*lamb/2*np.cos(psi)*np.cos(phi)
    ym = y - l0*lamb/2*np.cos(psi)*np.cos(phi)
    
    xc = max(x)
    yc = max(y)
    
    fig, ax = plt.subplots()
    ax.plot(x, y,'blue')
    ax.plot(2*xc-x, y,'blue')
    ax.plot(xp,yp,'red')
    ax.plot(2*xc-xp, yp,'red')
    ax.plot(xm,ym,'green')
    ax.plot(2*xc-xm, ym,'green')
    plt.title("D = "+str(int(D*1e6)/1e6))
    plt.axis('square')
    # plt.savefig("shape_E="+str(E)+"_l0^2="+str(round(l0**2))+
    #             "_D="+str(D)+".png",dpi=500)
    plt.show()


global D

Ds = np.linspace(Ds_i,Ds_f,100)
mus_opt = np.zeros_like(Ds)
lms_opt = np.zeros_like(Ds)
x_plus_dot = np.zeros_like(Ds)
y_init = np.zeros((6,sigma.size))
z = np.pi**2/4/E**2
D_star = 1-np.sqrt(1-z)

#D for cell shape 
Dx = 0.1
ix = np.where(abs(Ds - Dx) == min(abs(Ds-Dx)))[0]

i=0
for D in Ds:   
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
    f0 = 1 + p0**2/24/E**2
    x_plus_dot[i] = E*f0/lms_opt[i]/l0 - lms_opt[i]*l0*p0*np.cos(p0/2/E)/2
    
    y_init = sol
    
    if D==Ds[ix]:
        shape1(sol[0],sol[1],lms_opt[i],mus_opt[i],D)
    
    i += 1


plt.plot(Ds,x_plus_dot,'.-')

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
        dJ = dpsi**2/8/E**2*(1-2*s_star)
        
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
        
        
        return np.array([psi0-psi_max*s_star0, dpsi0 -(1-2*s_star0)* psi_max, I0,J0, dpsi1,
        I1 - lamb0*(1-D)/2 + (1+psi_max**2/8/E**2)*np.sin(psi_max*s_star0)/psi_max,
        
        2*s_star0*(psi_max*dpsi_max/4/lamb0/E**2 - psi_max**2/8/lamb0**2/E**2)+
        (1-1/lamb0**2)-2*J1/lamb0**2 + 2*mu0*(np.cos(psi_max*s_star0)/lamb0/psi_max*s_star0 \
        *dpsi_max*(1+psi_max**2/8/E**2)-np.sin(psi_max*s_star0)/lamb0/psi_max**2 \
        *dpsi_max*(1-psi_max**2/8/E**2)-1/2/lamb0*(1-D))])

    sol = intg.solve_bvp(func2,bc2,sigma2,y_init,tol=1e-8)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    s_star = sol.y[4]
    lamb = sol.y[5]
    J = sol.y[6]
    
    
    return [psi,dpsi,I,mu,s_star,lamb,J]


def shape2(psi,dpsi,lamb,mu,s_star,D):
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    sigma3 = np.linspace(0,s_star,500)
    ddpsi = 4*mu*E*E*np.sin(psi)*\
    (dpsi**2/8/E**2/(1-2*s_star)**2 - 1)*(1-2*s_star)**2/(1+mu*np.cos(psi))
    
    phi1 = psi_max/2/E*np.ones_like(sigma3)
    phi2 = dpsi/2/E/(1-2*s_star)
    phi = np.concatenate((phi1,phi2),axis=None)
    
    f = 1+dpsi**2/24/E**2/(1-2*s_star)**2
    g = ddpsi/12/E**2/(1-2*s_star)**2
    
    f_max = 1 + psi_max**2/24/E**2
    
    Icos = np.sin(psi_max*s_star)/psi_max  
    Isin = (1-np.cos(psi_max*s_star))/psi_max  
    
    dx = (f*np.cos(psi) - g*np.sin(psi))*(1-2*s_star)
    dy = (f*np.sin(psi) - g*np.cos(psi))*(1-2*s_star)
    
    x1 = (f_max*np.sin(psi_max*sigma3)/psi_max)/lamb*E/l0
    y1 = (f_max*(1-np.cos(psi_max*sigma3))/psi_max)/lamb*E/l0
    x2 =  (f_max*Icos  + np.cumsum(dx)/nodes)/lamb*E/l0
    y2 =  (f_max*Isin  + np.cumsum(dy)/nodes)/lamb*E/l0
    
    x = np.concatenate((x1,x2),axis=None)
    y = np.concatenate((y1,y2),axis=None)
    
    psi_i = np.concatenate((psi_max*sigma3,psi),axis=None)
    
    xp = x - l0*lamb/2*np.sin(psi_i)*np.cos(phi)
    xm = x + l0*lamb/2*np.sin(psi_i)*np.cos(phi)
    yp = y + l0*lamb/2*np.cos(psi_i)*np.cos(phi)
    ym = y - l0*lamb/2*np.cos(psi_i)*np.cos(phi)
    
    xc = x[-1]
    yc = y[-1]
    xc2 = max(2*xc-x)
    
    print(xm.shape)
    # print(max(2*xc-xp))
    # print(max(xm))
    
    if 2*xc-xp[0]+1e-3<max(max(2*xc-xp),max(xm)):
        print("Intersection!!!")

    fig, ax = plt.subplots()
    
    ax.plot(x, y,'blue')
    ax.plot(2*xc-x, 2*yc-y,'blue')
    ax.plot(2*xc2-x, y,'blue')
    ax.plot(2*xc2-2*xc+x, 2*yc-y,'blue')
    
    ax.plot(xp,yp,'red')
    ax.plot(2*xc-xp, 2*yc-yp,'green')
    ax.plot(2*xc2-xp, yp,'red')
    ax.plot(2*xc2-2*xc+xp, 2*yc-yp,'green')
    
    ax.plot(xm,ym,'green')
    ax.plot(2*xc-xm, 2*yc-ym,'red')
    ax.plot(2*xc2-xm, ym,'green')
    ax.plot(2*xc2-2*xc+xm, 2*yc-ym,'red')
    
    plt.title("D = "+str(int(D*1e6)/1e6))
    # plt.xlim(np.min(xp)-1,np.max(xm)+1)
    # plt.ylim(np.min(ym)-1,np.max(yp)+1)
    plt.axis('square')
    # plt.savefig("shape_E="+str(E)+"_l0^2="+str(round(l0**2))+
    #             "_D="+str(D)+".png",dpi=500)
    plt.show()


print("-------------above D_trg------------")


Ds2 = np.linspace(D_trg,Ds_f,100)
mus_opt2 = np.zeros_like(Ds2)
lms_opt2 = np.zeros_like(Ds2)
s_opt2 = np.zeros_like(Ds2)
y_init = np.zeros((7,sigma2.size))

#D for cell shape
Dx = 0.2
ix = np.where(abs(Ds2 - Dx) == min(abs(Ds2-Dx)))[0]

i=0
for D in Ds2:   
    if i < 1:
        eps = np.sqrt(1 - 1.005*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma2)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma2) * np.pi
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma2)
        y_init[4] = np.zeros_like(sigma2) + 1e-1
        y_init[5] = 1.00*np.ones_like(sigma2)
        y_init[6] =  (eps*2/(1-np.pi**2/4/E**2)*np.pi)**2 \
                    *(np.pi*sigma2/2+np.sin(2*np.pi*sigma2)/4)/8/E**2
        

    sol  = solve2(y_init)
    
    lms_opt2[i] = np.mean(sol[5])
    mus_opt2[i] = np.mean(sol[3])
    s_opt2[i] = np.mean(sol[4])
    y_init = sol
    
    if D==Ds2[ix]:
        shape2(sol[0],sol[1],lms_opt2[i],mus_opt2[i],s_opt2[i],D)
    
    i += 1


plt.plot(Ds2,s_opt2,'.-')
plt.ylabel(r"$s^*$")
plt.xlabel("D")
plt.show()


np.save("data/s_E="+str(E)+"_l0^2="+str(l0_sq)+".npy",np.vstack((Ds2,s_opt2)))
    
plt.plot(Ds,lms_opt,'.-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,lms_opt2,'.-',label=r'above $D_{\Delta}$')
plt.axvline(x=D_star,linestyle='--',c='black',linewidth=2,label=r'$D^*$')
plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=2,label=r'$D_{\Delta}$')
plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.xlim(0,Ds_f)
# plt.ylim(1,1.015)
plt.legend()
plt.title("Using previous initial condition")
# plt.savefig("lambda_l0_"+str(round(l0**2))+".png",dpi=500)
plt.show()


plt.plot(Ds,mus_opt,'.-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,mus_opt2,'.-',label=r'above $D_{\Delta}$')
plt.axvline(x=D_star,linestyle='--',c='black',linewidth=2,label=r'$D^*$')
plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=2,label=r'$D_{\Delta}$')
plt.xlabel("D")
plt.ylabel(r"$\mu$")
# plt.ylim(0.0,0.04)
plt.legend()
plt.title("Using previous initial condition")
# plt.savefig("mu_l0_"+str(round(l0**2))+".png",dpi=500)
plt.show()


# plt.plot(Ds2,s_opt2,'o-')
# plt.show()


print("Execution time:",datetime.now() - startTime)