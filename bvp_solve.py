import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg

nodes = 10000
sigma = np.linspace(0,1,nodes)

#parameters
E = 15
D = 0.005


def func(sigma,y):
    dy0 = y[1]
    dy1 = 4*y[3]*E*E*np.sin(y[0])*(y[1]**2/8/E**2 - 1)/(1+y[3]*np.cos(y[0]))
    dy2 = np.cos(y[0])*(1+y[1]**2/8/E**2)
    dy3 = np.zeros_like(y[1])
    return np.vstack((dy0,dy1,dy2,dy3))
    
def bc(ya,yb):
    return np.array([ya[0],ya[2],yb[0],yb[2] - lamb*(1-D)])

def solve():
    eps = np.sqrt(1 - (1-D))
    psi_init = eps*2/(1-np.pi**2/4/E**2)* np.sin(np.pi*sigma)
    dpsi_init = eps*2/(1-np.pi**2/4/E**2)* np.cos(np.pi*sigma) * np.pi
    
    y_init = np.zeros((4,sigma.size))
    y_init[0] = psi_init
    y_init[1] = dpsi_init
    y_init[2] = np.cos(psi_init)*(1+dpsi_init**2/8/E**2)
    y_init[3] = np.pi**2/4/E**2/(1-np.pi**2/4/E**2)*np.ones_like(sigma) 
    
    sol = intg.solve_bvp(func,bc,sigma,y_init)
    psi = sol.y[0]
    dpsi = sol.y[1]
    I = sol.y[2]
    mu = sol.y[3]
    
    return [dpsi, np.mean(mu)]


def lag(sig,dpsi):
    return lamb + 1/lamb + dpsi[int(sig*nodes)]**2/8/lamb/E**2

def L(dpsi):
    # return intg.quad(lag,0,1,args=(dpsi),limit=100000,epsrel=1e-15)[0]
    return intg.quad(lag,0,1,args=(dpsi),limit=100000,epsrel=1e-15)[0]
 
def get_min(a,b,tol,max_iter=1000):
    
    for _ in range(max_iter):

        m = (a+b)/2
        a1 = (m+a)/2
        b1 = (m+b)/2
        
        Ls = np.array([L(a),L(a1),L(m),L(b1),L(b)])
        idx = np.where(np.min(Ls)==Ls)[0][0] 
        mi =  a + idx/4 * (b-a)
        
        if abs(b-a)<tol:
            return mi
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
            
    return mi



# global lamb

# l1, l2 = 1, 1.05


# lamb = l1 
# dpsi , mu = solve()
# get_min(l1,l2,1e-6)
    
    
# print("Optimal mu for minimum L = ",mu_opt)
# print("Optimal lambda for minimum L = ",lm_opt)


global lamb
lms = np.linspace(1,1.01,100)
Ls = np.zeros_like(lms)
mus = np.zeros_like(lms)

j = 0
for l in lms:
    lamb = l
    dpsi , mus[j] = solve()
    Ls[j] = L(dpsi)
    j+=1

idx = np.where(Ls == np.min(Ls))[0]
lm_opt = lms[idx]
mu_opt = mus[idx] 
    
    
print("Optimal mu for minimum L = ",mu_opt)
print("Optimal lambda for minimum L = ",lm_opt)

plt.plot(lms,Ls)
plt.ylim(2,2.001)
