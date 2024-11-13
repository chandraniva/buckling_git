import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib.patches import Polygon

# gamma_a = 2 # Apical surface tension
# gamma_b = 1  # Basal surface tension
gamma_l = 1 # Lateral surface tension
Gamma = 5 #(gamma_a + gamma_b)/gamma_l
delta = 2.414 #(gamma_a - gamma_b)/gamma_l
gamma_a = (Gamma+delta)*gamma_l/2
gamma_b = (Gamma-delta)*gamma_l/2


N = 30

def energy(x):
    ls=x[0:N]
    phis=x[N:2*N]
    psis=x[2*N:3*N]
    
    ls_roll = np.roll(ls,-1)
    phis_roll = np.roll(phis,-1)
    psis_roll = np.roll(psis,-1)
    
    dels = 2/(ls*np.cos(psis-phis)+\
            ls_roll*np.cos(psis-phis_roll))
    ds = ls*np.sin(psis-phis)-\
         ls_roll*np.sin(psis-phis_roll) 
         
    lp = 1/2*np.sqrt(ls**2+ ls_roll**2 - \
         2*ls*ls_roll*np.cos(phis_roll-phis)+ \
         4*dels*(dels+ds))
    lm = 1/2*np.sqrt(ls**2+ ls_roll**2 - \
         2*ls*ls_roll*np.cos(phis_roll-phis)+ \
         4*dels*(dels-ds))
        
    E = np.sum(gamma_a/gamma_l*lp + gamma_b/gamma_l*lm + ls)
    
    return E

def constraint1(x):
    psis=x[2*N:3*N]
    return psis[0]-psis[-1]

def constraint2(x):
    phis=x[N:2*N]
    return phis[0]-phis[-1]

def constraint3(x):
    ls=x[0:N]
    return ls[0]-ls[-1]

def constraint4(x):
    ls=x[0:N]
    phis=x[N:2*N]
    psis=x[2*N:3*N]
    
    ls_roll = np.roll(ls,-1)
    phis_roll = np.roll(phis,-1)
    psis_roll = np.roll(psis,-1)
    
    dels = 2/(ls*np.cos(psis-phis)+\
            ls_roll*np.cos(psis-phis_roll))
         
    y0 = 0
    for i in range(0,N):
        y0 += dels[i]*np.sin(psis[i])
        
    return y0
    

def shape(x,labl):
    ls=x[0:N]
    phis=x[N:2*N]
    psis=x[2*N:3*N]
    
    ls_roll = np.roll(ls,-1)
    phis_roll = np.roll(phis,-1)
    psis_roll = np.roll(psis,-1)
    
    dels = 2/(ls*np.cos(psis-phis)+\
            ls_roll*np.cos(psis-phis_roll))
         
    x0, y0 = 0, 0 
    for i in range(0,N):
        vertex1 = [x0+ls[i]/2*np.sin(phis[i]),y0-ls[i]/2*np.cos(phis[i])]
        vertex2 = [x0+dels[i]*np.cos(psis[i])+ls[(i+1)%N]/2*np.sin(phis[(i+1)%N]),\
                   y0+dels[i]*np.sin(psis[i])-ls[(i+1)%N]/2*np.cos(phis[(i+1)%N])]
        vertex3 = [x0+dels[i]*np.cos(psis[i])-ls[(i+1)%N]/2*np.sin(phis[(i+1)%N]),\
                   y0+dels[i]*np.sin(psis[i])+ls[(i+1)%N]/2*np.cos(phis[(i+1)%N])]
        vertex4 =  [x0-ls[i]/2*np.sin(phis[i]),y0+ls[i]/2*np.cos(phis[i])]
        vertices = np.array([vertex1,vertex2,vertex3,vertex4])
        plt.gca().add_patch(Polygon(vertices, closed=True, fill=True, facecolor='lightgrey', edgecolor='black'))
        plt.axis([-5,N*sigma0+5,-5,N*sigma0+5])
        
        x0 += dels[i]*np.cos(psis[i])
        y0 += dels[i]*np.sin(psis[i])
        
    plt.title(labl)
    plt.show()
        
        
con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': constraint3}
con4 = {'type': 'eq', 'fun': constraint4}
con = [con1, con2, con3, con4]

sigma0 = 1/np.sqrt(Gamma)
q = 2*np.pi/N/sigma0


ls_init = 1/sigma0*np.ones(N).flatten()
phis_init = np.zeros(N).flatten()
psis_init = np.zeros(N).flatten()

for i in range(N):  
    A = 1
    ls_init[i] = 1/sigma0 +A*(-1/4*q*delta*np.sin(q*sigma0*i) + (2+Gamma**2-2*delta**2)*\
                (4*q**2*np.sqrt(Gamma)*np.cos(q*sigma0*i)+q**3*(2+Gamma**2-delta**2)\
                  *np.sin(q*sigma0*i))/(16*Gamma*delta*(Gamma**2-delta**2)))
        
    psis_init[i] = A*(np.cos(q*sigma0*i) + q**2*delta**2*np.cos(q*sigma0*i)/8/Gamma - \
                    q*np.sin(q*sigma0*i)/2/np.sqrt(Gamma)-q**3*(2+Gamma**2-2*delta**2)*\
                    (q*(2+Gamma**2-delta**2)*np.cos(q*sigma0*i)-4*np.sqrt(Gamma)*np.sin(q*sigma0*i))\
                      /(32*Gamma**2*(Gamma**2-delta**2))) 
                    
    phis_init[i] = A*np.cos(q*i*sigma0) 


x0 = np.concatenate([ls_init, phis_init,psis_init])

    
bl = (0,None)
bphi = (0,None)
bpsi = (0,None)
bounds = [bl for _ in range(N)]+\
          [bphi for _ in range(N)]+\
          [bpsi for _ in range(N)]

shape(x0,"Initial configuration")
sol = optimize.minimize(energy, x0,constraints = con, 
                        bounds = bounds,options={'disp': True})

shape(sol.x,"Minimized configuration")

print("Initial energy =",energy(x0))
print("Minimized energy =",energy(sol.x))
