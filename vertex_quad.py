import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib.patches import Polygon
from scipy import interpolate

N = 31
n = int((N + 1)/2)

gamma = 5
 
Di = 0.0
Df = 0.5
D_steps = 20

def draw_quads(phis,psis,ls):
    dels = 2/(ls[:-1]*np.cos(psis[:-1]-phis[:-1])+
              ls[1:]*np.cos(psis[:-1]-phis[1:]))
    
    phi_end = -phis[-1]
    psi_end = -psis[-2]
    l_end = ls[-1]
    
    dels_last = 2/(ls[-1]*np.cos(psis[-1]-phis[-1])+
              l_end*np.cos(psis[-1]+phi_end))

    xc = 0
    yc = 0

    for i in range(0,n-1):
        
        r1 = [ls[i]*np.sin(phis[i])/2+xc,-ls[i]*np.cos(phis[i])/2+yc]
        
        r2 = [dels[i]*np.cos(psis[i])+ls[i+1]/2*np.sin(phis[i+1])+xc,
              dels[i]*np.sin(psis[i])-ls[i+1]/2*np.cos(phis[i+1])+yc]
        
        r3 = [dels[i]*np.cos(psis[i])-ls[i+1]/2*np.sin(phis[i+1])+xc,
              dels[i]*np.sin(psis[i])+ls[i+1]/2*np.cos(phis[i+1])+yc]
        
        r4 = [-ls[i]*np.sin(phis[i])/2+xc,ls[i]*np.cos(phis[i])/2+yc]
        
        vertices = np.array([r1,r2,r3,r4])
        
        plt.gca().add_patch(Polygon(vertices, closed=True, fill=True,
                                facecolor='lightgrey',edgecolor='black'))
        
        xc += dels[i]*np.cos(psis[i])
        yc += dels[i]*np.sin(psis[i])
        
    
    r1 = [ls[n-1]*np.sin(phis[n-1])/2+xc,-ls[n-1]*np.cos(phis[n-1])/2+yc]
    
    r2 = [dels_last*np.cos(psis[n-1])+l_end/2*np.sin(phi_end)+xc,
          dels_last*np.sin(psis[n-1])-l_end/2*np.cos(phi_end)+yc]
    
    r3 = [dels_last*np.cos(psis[n-1])-l_end/2*np.sin(phi_end)+xc,
          dels_last*np.sin(psis[n-1])+l_end/2*np.cos(phi_end)+yc]
    
    r4 = [-ls[n-1]*np.sin(phis[n-1])/2+xc,ls[n-1]*np.cos(phis[n-1])/2+yc]
    
    vertices = np.array([r1,r2,r3,r4])
    
    plt.gca().add_patch(Polygon(vertices, closed=True, fill=True,
                            facecolor='lightgrey',edgecolor='black'))
    
    plt.title("D="+str(D))
    plt.axis([-2,N/gamma/2+2,-2,5])
    plt.show()

def energy(x):
    phis = x[0:n]
    psis = x[n:2*n]
    ls = x[2*n:]
    
    dels = 2/(ls[:-1]*np.cos(psis[:-1]-phis[:-1])+
              ls[1:]*np.cos(psis[:-1]-phis[1:]))
    
    phi_end = -phis[-1]
    psi_end = -psis[-2]
    l_end = ls[-1]
    
    dels_last = 2/(ls[-1]*np.cos(psis[-1]-phis[-1])+
              l_end*np.cos(psis[-1]+phi_end))
            
            
    lps = np.sqrt(ls[:-1]**2+ls[1:]**2+4*dels**2 -
                  2*ls[:-1]*ls[1:]*np.cos(phis[:-1]-phis[1:]) -
                  4*ls[:-1]*dels*np.sin(phis[:-1]-psis[:-1]) +
                  4*ls[1:]*dels*np.sin(phis[1:]-psis[:-1]))/2
        
    lms = np.sqrt(ls[:-1]**2+ls[1:]**2+4*dels**2 -
                  2*ls[:-1]*ls[1:]*np.cos(phis[:-1]-phis[1:]) +
                  4*ls[:-1]*dels*np.sin(phis[:-1]-psis[:-1]) -
                  4*ls[1:]*dels*np.sin(phis[1:]-psis[:-1]))/2
    
    lps_last = np.sqrt(ls[-1]**2+l_end**2+4*dels_last**2 -
                  2*ls[-1]*l_end*np.cos(phis[-1]-phi_end) -
                  4*ls[-1]*dels_last*np.sin(phis[-1]-psis[-1]) +
                  4*l_end*dels_last*np.sin(phi_end-psis[-1]))/2
    
    lms_last = np.sqrt(ls[-1]**2+l_end**2+4*dels_last**2 -
                  2*ls[-1]*l_end*np.cos(phis[-1]-phi_end) +
                  4*ls[-1]*dels_last*np.sin(phis[-1]-psis[-1]) -
                  4*l_end*dels_last*np.sin(phi_end-psis[-1]))/2
        
    
    e = np.sum(ls[1:]) + np.sum(lps[1:])*gamma + np.sum(lms[1:])*gamma +\
        (lps[0]/2*gamma + lms[0]/2*gamma) +\
        (lps_last/2*gamma + lms_last/2*gamma)
    return e


def constraint1(x):
    psis = x[n:2*n]
    return psis[-1]


def constraint2(x, D):
    phis = x[0:n]
    psis = x[n:2*n]
    ls = x[2*n:]

    sigma0 = 1/np.sqrt(2*gamma)
    
    dels = 2/(ls[:-1]*np.cos(psis[:-1]-phis[:-1])+
              ls[1:]*np.cos(psis[:-1]-phis[1:]))
        
    phi_end = -phis[-1]
    psi_end = -psis[-2]
    l_end = ls[-1]
    
    dels_last = 2/(ls[-1]*np.cos(psis[-1]-phis[-1])+
              l_end*np.cos(psis[-1]+phi_end))
    
    x_last = np.sum(dels*np.cos(psis[:-1]))+dels_last*np.cos(psis[-1])
    
    return x_last - n*sigma0*(1-D)


def constraint3(x):
    psis = x[n:2*n]
    ls = x[2*n:]

    return ls[1]-ls[0]

def constraint4(x):
    psis = x[n:2*n]
    phis = x[0:n]
    
    return phis[0]+phis[1]

def constraint5(x):
    psis = x[n:2*n]
    
    return psis[0]
    


 
z = np.pi**2/4/(N/2)**2
D_star = 1-np.sqrt(1-z)

Ds = np.linspace(Di,Df,D_steps)
amps = np.zeros_like(Ds)
lms0 = np.zeros_like(Ds)

ii = 0

for D in Ds:
    if D<D_star:
        phi_init = np.zeros(n)
        psi_init = np.zeros(n)
        ls_init = np.sqrt(2*gamma)*np.ones(n)
        x0 = np.concatenate([phi_init,psi_init,ls_init])
        
    if D==0:
        draw_quads(phi_init,psi_init,ls_init)
        print("energy_init:",energy(x0))
        
    elif D>D_star and D<D_star+0.1:
        sigma = np.arange(n)/n
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/(N/2)**2)* np.sin(np.pi*sigma) 
        dpsi_init = eps*2/(1-np.pi**2/4/(N/2)**2)* np.cos(np.pi*sigma) * np.pi
        phi_init = dpsi_init/N
        ls_init = np.sqrt(2*gamma)*np.ones(n)
        
        x0 = np.concatenate([phi_init,psi_init,ls_init])
        
    
    # b = (-10,10)
    # bounds = [b for _ in range(n)]+[(0.0*l0,4*l0)]
    
    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'eq', 'fun': constraint2, 'args': (D,)}
    con3 = {'type': 'eq', 'fun': constraint3}
    con4 = {'type': 'eq', 'fun': constraint4}
    con5 = {'type': 'eq', 'fun': constraint5}
    
    con = [con1, con2, con3, con4, con5]
    
    
    sol = optimize.minimize(energy, x0, constraints=con)
    x_opt = sol.x
    print("D="+str(D)+"; energy="+str(energy(x_opt)))
    phis = x_opt[0:n]
    psis = x_opt[n:2*n]
    ls = x_opt[2*n:]
    
    dels = 2/(ls[:-1]*np.cos(psis[:-1]-phis[:-1])+
              ls[1:]*np.cos(psis[:-1]-phis[1:]))
    amp = np.sum(dels*np.sin(psis[:-1]))
    amps[ii] = amp

    draw_quads(phis,psis,ls)
    
    x0 = x_opt
    
    if D == Ds[-1]:
        plt.plot(ls[1:],'o-')
        plt.ylabel("lateral lengths")
        plt.show()
        
    lms = np.sqrt(ls[:-1]**2+ls[1:]**2+4*dels**2 -
                  2*ls[:-1]*ls[1:]*np.cos(phis[:-1]-phis[1:]) +
                  4*ls[:-1]*dels*np.sin(phis[:-1]-psis[:-1]) -
                  4*ls[1:]*dels*np.sin(phis[1:]-psis[:-1]))/2
    lms0[ii] = lms[0]
    
    ii+=1



plt.plot(Ds,amps,'.-')
plt.axvline(x=D_star,linestyle='--',c='red')
plt.xlabel("D")
plt.ylabel("amplitude")
plt.show()


plt.plot(Ds,lms0,'.-')
plt.axhline(y=0,linestyle='--',c='red')
plt.xlabel("D")
plt.ylabel("apical length")
plt.show()