import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()

nodes = 1000
sigma = np.linspace(0,1,nodes)
stepsigma = 1/nodes

nodes2 = 1000
sigma2 = np.linspace(0,1/2,nodes2)
stepsigma2 = 0.5/nodes2

#parameters
E = 100
l0 = np.sqrt(85.8)
Ds_i = 0
Ds_f = 0.3

#D for cell shape 
Dx1 = 0.1
Dx2 = 0.22

#intersection tolerance
tol=1e-3*3

    
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
    dy = (f*np.sin(psi) + g*np.cos(psi))/lamb*E/l0
    
    x =  np.cumsum(dx)*stepsigma 
    y =  np.cumsum(dy)*stepsigma
    
    xp = x - l0*lamb/2*np.sin(psi)*np.cos(phi)
    xm = x + l0*lamb/2*np.sin(psi)*np.cos(phi)
    yp = y + l0*lamb/2*np.cos(psi)*np.cos(phi)
    ym = y - l0*lamb/2*np.cos(psi)*np.cos(phi)
    
    xc = max(x)
    yc = max(y)
    
    fig, ax = plt.subplots()
    ax.set(adjustable='box', aspect='equal')

    
    ax.plot(x, y,'blue')
    ax.plot(2*xc-x, y,'blue')
    ax.plot(xp,yp,'red')
    ax.plot(2*xc-xp, yp,'red')
    ax.plot(xm,ym,'green')
    ax.plot(2*xc-xm, ym,'green')
    
    plt.xlim(-2,E/l0*2+2)
    plt.ylim(-4,8)
    
    plt.title(r"Below $D_{\Delta}$; D = "+str(int(D*1e6)/1e6))
    plt.show()
    
def intersect1(psi,dpsi,lamb,mu):
    ddpsi = 4*mu*E**2*np.sin(psi)/(1+mu*np.cos(psi))*(dpsi**2/8/E**2 - 1)
    phi = dpsi/2/E
    f = 1+dpsi**2/24/E**2
    g = ddpsi/12/E**2 
    
    dx = (f*np.cos(psi) - g*np.sin(psi))/lamb*E/l0
    dy = (f*np.sin(psi) + g*np.cos(psi))/lamb*E/l0
    
    x =  np.cumsum(dx)*stepsigma 
    y =  np.cumsum(dy)*stepsigma
    
    xp = x - l0*lamb/2*np.sin(psi)*np.cos(phi)
    xm = x + l0*lamb/2*np.sin(psi)*np.cos(phi)
    yp = y + l0*lamb/2*np.cos(psi)*np.cos(phi)
    ym = y - l0*lamb/2*np.cos(psi)*np.cos(phi)
    
    xc = max(x)
    yc = max(y)
    
    if max(xm)-tol>min(2*xc-xm):
        return True
    else:
        return False
    
def amplitude1(psi,dpsi,lamb,mu,D):
    ddpsi = 4*mu*E**2*np.sin(psi)/(1+mu*np.cos(psi))*(dpsi**2/8/E**2 - 1)
    f = 1+dpsi**2/24/E**2
    g = ddpsi/12/E**2 
    dx = (f*np.cos(psi) - g*np.sin(psi))/lamb*E/l0
    dy = (f*np.sin(psi) + g*np.cos(psi))/lamb*E/l0
    x =  np.cumsum(dx)/nodes 
    y =  np.cumsum(dy)/nodes
    
    xc = max(x)
    yc = max(y)
    
    return yc



global D

Ds = np.linspace(Ds_i,Ds_f,100)
mus_opt = np.zeros_like(Ds)
lms_opt = np.zeros_like(Ds)
amp1 = np.zeros_like(Ds)
x_plus_dot = np.zeros_like(Ds)
y_init = np.zeros((6,sigma.size))
z = np.pi**2/4/E**2
D_star = 1-np.sqrt(1-z)


ix = np.where(abs(Ds - Dx1) == min(abs(Ds-Dx1)))[0]
i_ints1 = []
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
    elif D>D_star and D<D_star+0.1:
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
    
    flag = intersect1(sol[0],sol[1],lms_opt[i],mus_opt[i])
    if flag == True:
        i_ints1.append(i)
    
    if D==Ds[ix]:
        shape1(sol[0],sol[1],lms_opt[i],mus_opt[i],D)
    
    amp1[i] = amplitude1(sol[0],sol[1],lms_opt[i],mus_opt[i],D)
    i += 1


plt.plot(Ds,x_plus_dot,'.-')
plt.xlabel("D")
plt.ylabel(r"$\dot{x}_+$")

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


def intersect2(psi,dpsi,lamb,mu,s_star,D):
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    sig3 = np.linspace(0,s_star,100)
    
    ddpsi = 4*mu*E*E*np.sin(psi)*\
    (dpsi**2/8/E**2/(1-2*s_star)**2 - 1)*(1-2*s_star)**2/(1+mu*np.cos(psi))
    
    phi1 = psi_max/2/E*np.ones_like(sig3)
    phi2 = dpsi/2/E/(1-2*s_star)
    phi = np.concatenate((phi1,phi2),axis=None)
    
    f = 1+dpsi**2/24/E**2/(1-2*s_star)**2
    g = ddpsi/12/E**2/(1-2*s_star)**2
    
    f_max = 1 + psi_max**2/24/E**2
    

    dx = (f*np.cos(psi) - g*np.sin(psi))*(1-2*s_star)
    dy = (f*np.sin(psi) + g*np.cos(psi))*(1-2*s_star)
    
    x1 = (f_max*np.sin(psi_max*sig3)/psi_max)/lamb*(E/l0)
    y1 = (f_max*(1-np.cos(psi_max*sig3))/psi_max)/lamb*(E/l0)
    
    x2 =  x1[-1] + (np.cumsum(dx)*stepsigma2)/lamb*(E/l0)
    y2 =  y1[-1] + (np.cumsum(dy)*stepsigma2)/lamb*(E/l0)
    
    x = np.concatenate((x1,x2),axis=None)
    y = np.concatenate((y1,y2),axis=None)
    
    psi_i = np.concatenate((psi_max*sig3,psi),axis=None)
    
    xp = x - l0*lamb/2*np.sin(psi_i)*np.cos(phi)
    xm = x + l0*lamb/2*np.sin(psi_i)*np.cos(phi)
    yp = y + l0*lamb/2*np.cos(psi_i)*np.cos(phi)
    ym = y - l0*lamb/2*np.cos(psi_i)*np.cos(phi)
    
    xc = x[-1]
    yc = y[-1]
    xc2 = 2*xc    
    # print(D,max(max(2*xc-xp),max(xm)),min(min(2*xc2-xm),min(2*xc2-2*xc+xp)))
    if max(max(2*xc-xp),max(xm))-tol>min(min(2*xc2-xm),min(2*xc2-2*xc+xp)):
        return True
    else:
        return False
    
    
    
def amplitude2(psi,dpsi,lamb,mu,s_star,D):
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    sig3 = np.linspace(0,s_star,100)
    
    ddpsi = 4*mu*E*E*np.sin(psi)*\
    (dpsi**2/8/E**2/(1-2*s_star)**2 - 1)*(1-2*s_star)**2/(1+mu*np.cos(psi))
    
    f = 1+dpsi**2/24/E**2/(1-2*s_star)**2
    g = ddpsi/12/E**2/(1-2*s_star)**2
    
    f_max = 1 + psi_max**2/24/E**2
    
    
    dx = (f*np.cos(psi) - g*np.sin(psi))*(1-2*s_star)
    dy = (f*np.sin(psi) + g*np.cos(psi))*(1-2*s_star)
    
    x1 = (f_max*np.sin(psi_max*sig3)/psi_max)/lamb*(E/l0)
    y1 = (f_max*(1-np.cos(psi_max*sig3))/psi_max)/lamb*(E/l0)
    
    x2 =  x1[-1] + (np.cumsum(dx)*stepsigma2)/lamb*(E/l0)
    y2 =  y1[-1] + (np.cumsum(dy)*stepsigma2)/lamb*(E/l0)
    
    x = np.concatenate((x1,x2),axis=None)
    y = np.concatenate((y1,y2),axis=None)
    
    xc = x[-1]
    yc = y[-1]
    
    return 2*yc

    
def shape2(psi,dpsi,lamb,mu,s_star,D):
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    print("psi_max=",psi_max)
    sig3 = np.linspace(0,s_star,100)
    
    ddpsi = 4*mu*E*E*np.sin(psi)*\
    (dpsi**2/8/E**2/(1-2*s_star)**2 - 1)*(1-2*s_star)**2/(1+mu*np.cos(psi))
    
    phi1 = psi_max/2/E*np.ones_like(sig3)
    phi2 = dpsi/2/E/(1-2*s_star)
    phi = np.concatenate((phi1,phi2),axis=None)
    
    f = 1+dpsi**2/24/E**2/(1-2*s_star)**2
    g = ddpsi/12/E**2/(1-2*s_star)**2
    
    f_max = 1 + psi_max**2/24/E**2
    

    dx = (f*np.cos(psi) - g*np.sin(psi))*(1-2*s_star)
    dy = (f*np.sin(psi) + g*np.cos(psi))*(1-2*s_star)
    
    x1 = (f_max*np.sin(psi_max*sig3)/psi_max)/lamb*(E/l0)
    y1 = (f_max*(1-np.cos(psi_max*sig3))/psi_max)/lamb*(E/l0)
    
    x2 =  x1[-1] + (np.cumsum(dx)*stepsigma2)/lamb*(E/l0)
    y2 =  y1[-1] + (np.cumsum(dy)*stepsigma2)/lamb*(E/l0)
    
    x = np.concatenate((x1,x2),axis=None)
    y = np.concatenate((y1,y2),axis=None)
    
    psi_i = np.concatenate((psi_max*sig3,psi),axis=None)
    
    xp = x - l0*lamb/2*np.sin(psi_i)*np.cos(phi)
    xm = x + l0*lamb/2*np.sin(psi_i)*np.cos(phi)
    yp = y + l0*lamb/2*np.cos(psi_i)*np.cos(phi)
    ym = y - l0*lamb/2*np.cos(psi_i)*np.cos(phi)
    

    
    xc = x[-1]
    yc = y[-1]
    xc2 = 2*xc
    
    
    fig, ax = plt.subplots()
    ax.set(adjustable='box', aspect='equal')
    
    
    ax.plot(x, y,'blue')
    # ax.plot(2*xc-x, 2*yc-y,'blue')
    # ax.plot(2*xc2-x, y,'blue')
    # ax.plot(2*xc2-2*xc+x, 2*yc-y,'blue')
    
    ax.plot(xp,yp,'red')
    # ax.plot(2*xc-xp, 2*yc-yp,'green')
    # ax.plot(2*xc2-xp, yp,'red')
    # ax.plot(2*xc2-2*xc+xp, 2*yc-yp,'green')
    
    ax.plot(xm,ym,'green')
    # ax.plot(2*xc-xm, 2*yc-ym,'red')
    # ax.plot(2*xc2-xm, ym,'green')
    # ax.plot(2*xc2-2*xc+xm, 2*yc-ym,'red')
    
    plt.title(r"Above $D_{\Delta}$; D = "+str(int(D*1e6)/1e6))
    
    # plt.xlim(-2,E/l0*2+2)
    # plt.ylim(-4,8)
    # plt.savefig("shape_D="+str(int(D*1e3)/1e3)+"_E="+str(E)+"_l0^2="+
    #             str(round(l0**2*1000)/1000)+".png",dpi=500)
    
    # plt.xlim(3.265,3.28)
    # plt.ylim(1.245,1.26)
    # plt.savefig("zoomed_shape_D="+str(int(D*1e3)/1e3)+"_E="+str(E)+"_l0^2="+
    #             str(round(l0**2*1000)/1000)+".png",dpi=500)
    
    
    plt.show()
    
    plt.plot(psi)
    plt.title("psi")
    plt.show()
    
    plt.plot(phi)
    plt.title("phi")
    plt.show()
    



print("-------------above D_trg------------")


Ds2 = np.linspace(D_trg,Ds_f,200)
mus_opt2 = np.zeros_like(Ds2)
lms_opt2 = np.zeros_like(Ds2)
s_opt2 = np.zeros_like(Ds2)
y_init = np.zeros((7,sigma2.size))
amp2 = np.zeros_like(Ds2)


ix = np.where(abs(Ds2 - Dx2) == min(abs(Ds2-Dx2)))[0]
i_ints = []
i=0
for D in Ds2:   
    if i < 3:
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma2)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma2) * np.pi
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma2)
        y_init[4] = np.zeros_like(sigma2) + 0.1#1e-1
        y_init[5] = 1.10*np.ones_like(sigma2)
        y_init[6] =  (eps*2/(1-np.pi**2/4/E**2)*np.pi)**2 \
                    *(np.pi*sigma2/2+np.sin(2*np.pi*sigma2)/4)/8/E**2
        

    sol  = solve2(y_init)
    
    lms_opt2[i] = np.mean(sol[5])
    mus_opt2[i] = np.mean(sol[3])
    s_opt2[i] = np.mean(sol[4])
    y_init = sol

    amp2[i] = amplitude2(sol[0],sol[1],lms_opt2[i],mus_opt2[i],s_opt2[i],D)
    flag = intersect2(sol[0],sol[1],lms_opt2[i],mus_opt2[i],s_opt2[i],D)
    if flag == True:
        i_ints.append(i)
        
    if D==Ds2[ix]:
        shape2(sol[0],sol[1],lms_opt2[i],mus_opt2[i],s_opt2[i],D)
    
    i += 1

D_intsct2 = Ds_f + 1e2
if i_ints:
    D_intsct2 = Ds2[min(i_ints)]

D_intsct1 = Ds_f + 1e2
if i_ints1:
    D_intsct1 = Ds[min(i_ints1)]

D_intsct= D_intsct2
if D_intsct1<D_trg:   
    D_intsct = D_intsct1



np.save("data/lms_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy",np.vstack((Ds2,lms_opt2)))
np.save("data/mus_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy",np.vstack((Ds2,mus_opt2)))
np.save("data/intsct_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy",np.array(D_intsct))
np.save("data/amp_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy",np.vstack((Ds2,amp2)))
np.save("data/amp1_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy",np.vstack((Ds,amp1)))
np.save("data/s_star_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
        "_tol="+str(tol)+".npy",np.vstack((Ds2,s_opt2)))


plt.plot(Ds2,s_opt2,'.-')
plt.ylabel(r"$s^*$")
plt.xlabel("D")
plt.show()
 

plt.plot(Ds,lms_opt,'.-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,lms_opt2,'.-',label=r'above $D_{\Delta}$')
plt.axvline(x=D_star,linestyle='--',c='black',linewidth=2,label=r'$D^*$')
plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=2,label=r'$D_{\Delta}$')
if i_ints:
    plt.axvline(x=D_intsct,linestyle='--',c='green',linewidth=2,
                label='Self-intersection')
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
if i_ints:
    plt.axvline(x=D_intsct,linestyle='--',c='green',linewidth=2,
            label='Self-intersection')
plt.xlabel("D")
plt.ylabel(r"$\mu$")
# plt.ylim(0.0,0.04)
plt.legend()
plt.title("Using previous initial condition")
# plt.savefig("mu_l0_"+str(round(l0**2))+".png",dpi=500)
plt.show()


plt.plot(Ds,amp1,'.-')
plt.plot(Ds2,amp2,'.-')
if i_ints:
    plt.axvline(x=D_intsct,linestyle='--',c='green',linewidth=2,
            label='Self-intersection')
plt.xlabel("D")
plt.ylabel("Amplitude")
plt.title(r"$\Xi$="+str(E)+str("$; l_0^2=$")+str(round(l0**2*10000)/10000))
plt.show()


plt.plot(Ds2[1:],amp2[1:]-amp2[:-1],'.-')
plt.show()

amp_diff = np.abs(amp2[1:]-amp2[:-1])

print("Bifurcation at D=",Ds2[np.where(amp_diff==max(amp_diff))[0]])
print(Ds2[0],D_intsct)


print("Execution time:",datetime.now() - startTime)