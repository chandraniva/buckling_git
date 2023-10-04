import numpy as np

def L(x):
    return x**2 

def get_min(a,b,tol):
    
    m = (a+b)/2
    a1 = (m+a)/2
    b1 = (m+b)/2
    
    m0 = m + 2*tol  
    
    while np.abs(m-m0)>tol:
        
        m0 = m
        
        if L(a1)<L(a) and L(a1)<L(m):
            b = m
            m = get_min(a,b,tol)
        elif L(b1)<L(b) and L(b1)<L(m):
            a = m
            m = get_min(a,b,tol)
        elif L(m)<L(a1) and L(m)<L(b1):
            m = get_min(a1,b1,tol)
            
    return m
            
        
res = get_min(-2,3,0.01)
print(res)
    
    
    