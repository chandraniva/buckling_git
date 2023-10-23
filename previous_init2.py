import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.optimize as opt
from datetime import datetime
from scipy import interpolate

startTime = datetime.now()

nodes = 2000
sigma = np.linspace(0,1,nodes)

nodes2 = 1000
sigma2 = np.linspace(0,1/2,nodes2)

#parameters
E = 15
l0 = np.sqrt(15)
Ds_i = 0
Ds_f = 0.1
l1, l2 = 1, 1.015


    
def solve(lamb,y_init):
    def func(sigma,y):
        dy0 = y[1]
        dy1 = 4*y[3]*E*E*np.sin(y[0])*(y[1]**2/8/E**2 - 1)/(1+y[3]*np.cos(y[0]))
        dy2 = np.cos(y[0])*(1+y[1]**2/8/E**2)
        dy3 = np.zeros_like(y[1])
        return np.vstack((dy0,dy1,dy2,dy3))
        
    def bc(ya,yb):
        return np.array([ya[0],ya[2],yb[0],yb[2] - lamb*(1-D)])
    
    
    sol = intg.solve_bvp(func,bc,sigma,y_init)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    
    return np.array([psi,dpsi,I,mu])


def lag(sig,dpsi,lamb):
    return lamb + 1/lamb + dpsi[int(sig*nodes)]**2/8/lamb/E**2

def L(lamb,sol):
    dpsi = sol[1]
    return intg.quad(lag,0,1,args=(dpsi,lamb),limit=100000,epsrel=1e-5)[0]
 
def get_min(a,b,tol,y_init,max_iter=1000):
    
    for _ in range(max_iter):

        m = (a+b)/2
        a1 = (m+a)/2
        b1 = (m+b)/2
        
        a_sol = solve(a,y_init)
        a1_sol = solve(a1,y_init)
        m_sol= solve(m,y_init)
        b1_sol = solve(b1,y_init)
        b_sol = solve(b,y_init)
        
        Ls = np.array([L(a,a_sol),L(a1,a1_sol),L(m,m_sol),L(b1,b1_sol), 
                       L(b,b_sol)])
        idx = np.where(np.min(Ls)==Ls)[0][0]
        lm_min =  a + idx/4 * (b-a)
        
        if abs(b-a)<tol:
            sol_min = solve(lm_min,y_init)
            return [lm_min, sol_min]
        
        if idx == 0:
            b = a1
        elif idx == 1:
            b = m
        elif idx == 2:
            a = a1
            b = b1
        elif idx == 3:
            a = m
        elif idx == 4:
            a = b1
    
    print("---------- No convergence of solution ----------")
    
    
global D


Ds = np.linspace(Ds_i,Ds_f,40)
mus_opt = np.zeros_like(Ds)
lms_opt = np.zeros_like(Ds)
x_plus_dot = np.zeros_like(Ds)
y_init = np.zeros((4,sigma.size))


i=0
for D in Ds:   
    print(D)
    if i < 4:
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma) * np.pi
        
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma) 
        
    lms_opt[i], sol  = get_min(l1,l2,1e-6,y_init)
    mus_opt[i] = np.mean(sol[3])
    dpsi_opt = sol[1]
    p0 = dpsi_opt[0]
    f0 = 1 + p0**2/24/l0**2
    x_plus_dot[i] = E*f0/lms_opt[i]/l0-lms_opt[i]*l0*p0*np.cos(p0/2/l0)/2
    
    y_init = sol
    
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


            
def solve2(lamb,y_init):
           
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
        
        return np.array([psi0-psi_max*s_star0, dpsi0 - (1-s_star0)*psi_max, I0, dpsi1,
        I1 - lamb*(1-D)/2 + (1+psi_max**2/8/E**2)*np.sin(psi_max*s_star0)/psi_max])

    sol = intg.solve_bvp(func2,bc2,sigma2,y_init)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    s_star = sol.y[4]
    
    
    return [psi,dpsi,I,mu,s_star]


def lag2(sig,dpsi,lamb,s_star):
    return lamb + 1/lamb + dpsi[int(sig*nodes2)]**2/8/lamb/E**2/(1-2*s_star)**2


def L2(lamb,sol):
    dpsi = sol[1]
    s_star = np.mean(sol[4])
    
    roots = np.roots([l0**2 * lamb/16/E**3, 1/24/lamb/E**2, -l0**2 *lamb/2/E,
                      1/lamb])
    psi_max = roots[-1]
    
    return 2*s_star*(lamb+1/lamb+psi_max**2/8/lamb/E**2) + \
    2*(1-2*s_star)*intg.quad(lag2,0,1/2,args=(dpsi,lamb,s_star),limit=100000,epsrel=1e-5)[0]
 
    
def get_min2(a,b,tol,y_init,max_iter=1000):
    
    for _ in range(max_iter):

        m = (a+b)/2
        a1 = (m+a)/2
        b1 = (m+b)/2
        
        a_sol = solve2(a,y_init)
        a1_sol = solve2(a1,y_init)
        m_sol= solve2(m,y_init)
        b1_sol = solve2(b1,y_init)
        b_sol = solve2(b,y_init)
        
        
        
        Ls = np.array([L2(a,a_sol),L2(a1,a1_sol),L2(m,m_sol),L2(b1,b1_sol), 
                       L2(b,b_sol)])
        idx = np.where(np.min(Ls)==Ls)[0][0] 
        lm_min =  a + idx/4 * (b-a)
        
        if abs(b-a)<tol:
            sol_min = solve2(lm_min,y_init)
            return [lm_min, sol_min]
        
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
    
    print(" ---------- No convergence of solution ----------")


print("-------------Bisection method-------------")




Ds2 = np.linspace(D_trg,Ds_f,20)
mus_opt2 = np.zeros_like(Ds2)
lms_opt2 = np.zeros_like(Ds2)
s_opt2 = np.zeros_like(Ds2)
y_init = np.zeros((5,sigma2.size))

i=0
for D in Ds2:   
    print(D)
    if i == 0:
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma2)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma2) * np.pi
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma2)
        y_init[4] = np.zeros_like(sigma2)

    lms_opt2[i], sol  = get_min2(l1,l2,1e-6,y_init)
    mus_opt2[i] = np.mean(sol[3])
    s_opt2[i] = np.mean(sol[4])
    y_init = sol
    i += 1

print("--------------- brute force ---------------------")


lms = np.linspace(l1,l2,100)
Ds3 = np.linspace(D_trg,Ds_f,10)
mus_opt3 = np.zeros_like(Ds3)
lms_opt3 = np.zeros_like(Ds3)
y_init = np.zeros((5,sigma2.size))

i=0
for D in Ds3:   
    print(D)
    if i == 0:
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma2)
        dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma2) * np.pi
        
        y_init[0] = psi_init
        y_init[1] = dpsi_init
        y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
        y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma2)
        y_init[4] = np.zeros_like(sigma2)

    Ls = np.zeros_like(lms)
    mus = np.zeros_like(lms)
    
    j = 0
    for l in lms:
        sol = solve2(l,y_init)
        mus[j] = np.mean(sol[3])
        Ls[j] = L2(l,sol)
        j+=1
    
    idx = np.where(Ls == np.min(Ls))[0][0]
    lms_opt3[i] = lms[idx]
    mus_opt3[i] = mus[idx] 
    y_init = sol
    i += 1
    
    
plt.plot(Ds,lms_opt,'o-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,lms_opt2,'o-',label=r'above $D_{\Delta}$: bisection')
# plt.plot(Ds3,lms_opt3,'o-',label=r'above $D_{\Delta}$: brute')
plt.xlabel("D")
plt.ylabel(r"$\Lambda$")
plt.ylim(l1,l2)
plt.legend()
plt.title("Using previous initial condition")
plt.savefig("lambda_pinit.png",dpi=500)
plt.show()


plt.plot(Ds,mus_opt,'o-',label=r'below $D_{\Delta}$')
plt.plot(Ds2,mus_opt2,'o-',label=r'above $D_{\Delta}$: bisection')
# plt.plot(Ds3,mus_opt3,'o-',label=r'above $D_{\Delta}$: brute')
plt.xlabel("D")
plt.ylabel(r"$\mu$")
plt.ylim(0.01,0.02)
plt.legend()
plt.title("Using previous initial condition")
plt.savefig("mu_pinit.png",dpi=500)
plt.show()


plt.plot(Ds2,s_opt2,'o-')
plt.show()


print("Execution time:",datetime.now() - startTime)