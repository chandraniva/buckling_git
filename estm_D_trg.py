import numpy as np
import matplotlib.pyplot as plt

Es = np.linspace(8,100,50)
l0 = np.sqrt(10)
pi = np.pi
D_trgs = np.zeros_like(Es)

i = 0
for E in Es:
    xi = pi/E
    r = E/l0**2
    z = pi/(4*E**2)
    eps0 = 2*np.sqrt((4-xi**2)/(32-14*xi**2+xi**4))
    psi0 = 2/np.sqrt(1-z)
    D_star = 1 - np.sqrt(1-z)
    
    roots = np.roots([eps0**3*psi0*pi/r/(1-D_star)**2-eps0*psi0*pi/(1-D_star)**3,
                      eps0**2*psi0**2*pi*z/6,-eps0*psi0*pi/2/r/(1-D_star)**2,1])
    D_trgs[i] = (np.real(roots[-1]))**2 + D_star
    
    i+=1
    
    
plt.plot(Es,D_trgs)
plt.loglog()
plt.show()