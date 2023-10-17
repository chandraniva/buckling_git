import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate



E = 15
l0 = np.sqrt(20)
nodes = 1000
sigma2 = np.linspace(0,1/2,nodes)

            
            
def solve2(lamb):
           
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    
    def func2(sigma2,y):
        
        psi = y[0]
        dpsi = y[1]
        I = y[2]
        mu = y[3]
        s_star = y[4]
        
        dpsi = dpsi
        ddpsi = 4*mu*E*E*np.sin(psi)*(dpsi**2/8/E**2/(1-2*s_star)**2 - 1)\
              *(1-2*s_star)**2/(1+mu*np.cos(psi))
        dI = np.cos(psi)*(1+dpsi**2/8/E**2/(1-2*s_star)**2)*(1-2*s_star)
        dmu = np.zeros_like(psi)
        ds_star = np.zeros_like(psi)
        
        return np.vstack((dpsi,ddpsi,dI,dmu,ds_star))
        
    def bc2(ya,yb):
        
        psi0 = ya[0]
        dpsi0 = ya[1]
        I0 = ya[2]
        s_star0 = ya[4]

        dpsi1 = yb[1]
        I1 = yb[2]
        s_star1 = yb[4]
        
        return np.array([psi0-psi_max*s_star0, dpsi0 - 1*psi_max, I0, dpsi1,
        I1 - lamb*(1-D)/2 - (1+psi_max**2/8/E**2)*np.sin(psi_max*s_star0)/psi_max])

    
    eps = np.sqrt(1 - lamb*(1-D))
    #eps = eps0*(D-D0)
    
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


def lag(sig,dpsi,lamb):
    return lamb + 1/lamb + dpsi[int(sig*nodes)]**2/8/lamb/E**2

def L2(dpsi,lamb,s_star):
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    
    return 2*s_star*(lamb+1/lamb+psi_max**2/8/lamb/E**2) +  \
    2*intg.quad(lag,s_star,1/2,args=(dpsi,lamb),limit=5000,epsrel=1e-10)[0]
 
    
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
        
        Ls = np.array([L2(a_dpsi,a,a_star),L2(a1_dpsi,a1,a1_star),L2(m_dpsi,m,m_star),
                       L2(b1_dpsi,b1,b1_star), L2(b_dpsi,b,b_star)])
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

lms = np.linspace(1,1.02,50)
Ds = np.linspace(0.06,0.1,20)
# Ds = [0.07]
mus_opt = np.zeros_like(Ds)
lms_opt = np.zeros_like(Ds)


i=0
for D in Ds:   
    print(D)
    Ls = np.zeros_like(lms)
    mus = np.zeros_like(lms)
    
    j = 0
    for l in lms:
        dpsi , mus[j], s_star = solve2(l)
        Ls[j] = L2(dpsi,l,s_star)
        j+=1
    
    idx = np.where(Ls == np.min(Ls))[0]
    lms_opt[i] = lms[idx]
    mus_opt[i] = mus[idx] 
    i += 1
    
    # plt.plot(lms,Ls,'.-')
    # plt.show()
    

plt.plot(Ds,lms_opt,'o-')
plt.xlabel("D")
plt.ylabel("lambda")
plt.show()

plt.plot(Ds,mus_opt,'o-')
plt.xlabel("D")
plt.ylabel("mu")
plt.show()

    
 