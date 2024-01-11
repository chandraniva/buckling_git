import numpy as np
import matplotlib.pyplot as plt

E = 99
l0 = np.sqrt(85.7)
Ds_i = 0
Ds_f = 0.21

nodes2 = 1000
sigma2 = np.linspace(0,1/2,nodes2)
stepsigma2 = 0.5/nodes2

s_star = 0.499
lamb = 0.8

sig2 = s_star + (sigma2)*(1-2*s_star)
sig3 = np.linspace(0,s_star,100)

def shape_dpsi(dpsi,psi_max,s_star,lamb):
    print("😳")
          
    psi = psi_max*s_star+np.cumsum(dpsi)*stepsigma2
          
    ddpsi = -2*psi_max*np.ones_like(dpsi)

    sig3 = np.linspace(0,s_star,100)
    
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
    
    plt.plot(sig2,psi)
    plt.plot(sig3,sig3*psi_max)
    plt.title("psi")
    plt.show()
    
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
    # ax.plot(x2, y2,'red')
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
    
    plt.show()
    


roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                  1/lamb])
psi_max = roots[-1]
print("psi_max=",psi_max)


dpsi = psi_max/(s_star-0.5)*(sig2-0.5)*(1-2*s_star)
plt.plot(sig2,dpsi/(1-2*s_star))
plt.plot(sig3,np.ones_like(sig3)*psi_max)
plt.title("dpsi: Input")
plt.show()


shape_dpsi(dpsi,psi_max,s_star,lamb)