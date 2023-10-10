import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate


#parameters
D_trg = 0.06

E = 15
l0 = np.sqrt(20)
nodes = 1000
sigma2 = np.linspace(0,1/2,nodes)

            
            
def solve2(lamb):
           
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    
    def func2(sigma,y):
        dy0 = y[1]
        dy1 = 4*y[3]*E*E*np.sin(y[0])*(y[1]**2/8/E**2/(1-2*y[4])**2 - 1)\
              *(1-2*y[4])**2/(1+y[3]*np.cos(y[0]))
        dy2 = np.cos(y[0])*(1+y[1]**2/8/E**2/(1-2*y[4])**2)*(1-2*y[4])
        dy3 = np.zeros_like(y[1])
        dy4 = np.zeros_like(y[1])
        return np.vstack((dy0,dy1,dy2,dy3,dy4))
        
    def bc2(ya,yb):
        return np.array([ya[0]-psi_max*ya[4],ya[1] - psi_max, ya[2],
        yb[1],yb[2] - lamb*(1-D)/2 - (1+psi_max**2/8/E**2)*np.sin(psi_max*ya[4])/psi_max])

    
    eps = np.sqrt(1 - 1*(1-D))
    psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma2)
    dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma2) * np.pi
    
    y_init = np.zeros((5,sigma2.size))
    y_init[0] = psi_init
    y_init[1] = dpsi_init
    y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
    y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma2)
    y_init[4] = np.zeros_like(sigma2)
    
    sol = intg.solve_bvp(func2,bc2,sigma2,y_init)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    s_star = sol.y[4]
    
    return [dpsi, np.mean(mu), s_star]



global D
D = 0.08
l1, l2 = 1, 1.03
Ds2 = np.linspace(D_trg,0.1,100)

dpsi,mu,s_star = solve2(1.006)
print(dpsi,mu)
plt.plot(sigma2,s_star)

# i=0
# for D in Ds2:  
#     lms_opt[i], mus_opt[i], dpsi_opt = get_min(l1,l2,1e-8)
#     p0 = dpsi_opt[0]
#     f0 = 1 + p0**2/24/l0**2
#     x_plus_dot[i] = E* f0/lms_opt[i]/l0-lms_opt[i]/2*l0*p0*np.cos(p0/2/l0) 
#     i += 1
    
 