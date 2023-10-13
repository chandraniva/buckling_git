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
    
    return [dpsi, np.mean(mu), np.mean(s_star)]


def lag(sig,dpsi,lamb,s_star):
    return lamb + 1/lamb + dpsi[int(sig*nodes)]**2/8/lamb/E**2

def L(dpsi,lamb,s_star):
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    
    return (1-2*s_star)*(2*s_star*(lamb+1/lamb+psi_max**2/8/lamb/E**2) + \
    2*intg.quad(lag,s_star,1/2,args=(dpsi,lamb,s_star),limit=5000,epsrel=1e-10)[0])
 
    
def get_min2(a,b,tol,max_iter=1000):
    
    for _ in range(max_iter):

        m = (a+b)/2
        a1 = (m+a)/2
        b1 = (m+b)/2
        
        a_dpsi , a_mu, a_star = solve2(a)
        a1_dpsi , a1_mu, a1_star = solve2(a1)
        m_dpsi , m_mu, m_star = solve2(m)
        b1_dpsi , b1_mu, b1_star = solve2(b1)
        b_dpsi , b_mu, b_star = solve2(b)
        
        Ls = np.array([L(a_dpsi,a,a_star),L(a1_dpsi,a1,a1_star),L(m_dpsi,m,m_star),
                       L(b1_dpsi,b1,b1_star), L(b_dpsi,b,b_star)])
        idx = np.where(np.min(Ls)==Ls)[0][0] 
        lm_min =  a + idx/4 * (b-a)
        
        if abs(b-a)<tol:
            dpsi_min, mu_min, s_star_min = solve2(lm_min)
            return [lm_min, mu_min, dpsi_min]
        
        if idx == 0:
            b = a1
        elif idx==1:
            b = m
        elif idx == 2:
            a = a1
            b = b1
        elif idx == 3:
            a = m
        elif idx == 4:
            a = b1
    
    dpsi_min, mu_min, s_star_min = solve2(lm_min)
    return [lm_min, mu_min, dpsi_min]

global D
D = 0.06
l1, l2 = 1, 1.03
Ds2 = np.linspace(D_trg,0.1,100)
mus_opt2 = np.zeros_like(Ds2)
lms_opt2 = np.zeros_like(Ds2)
x_plus_dot2 = np.zeros_like(Ds2)

dpsi,mu,s_star = solve2(1.006)

i=0
for D in Ds2:   
    print(D)
    lms_opt2[i], mus_opt2[i], dpsi_opt = get_min2(l1,l2,1e-8)
    i += 1


plt.plot(Ds2,lms_opt2,'o-')
plt.xlabel("D")
plt.ylabel("lambda")
# plt.savefig("lambda.png",dpi=500)
plt.show()

plt.plot(Ds2,mus_opt2,'o-')
plt.xlabel("D")
plt.ylabel("mu")
# plt.savefig("mu.png",dpi=500)
plt.show()

# i=0
# for D in Ds2:  
#     lms_opt[i], mus_opt[i], dpsi_opt = get_min(l1,l2,1e-8)
#     p0 = dpsi_opt[0]
#     f0 = 1 + p0**2/24/l0**2
#     x_plus_dot[i] = E* f0/lms_opt[i]/l0-lms_opt[i]/2*l0*p0*np.cos(p0/2/l0) 
#     i += 1
    
 