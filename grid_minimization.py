import numpy as np

def L(x):
    return x**4-x**2 

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
        
            
 
res = get_min(0,3,0.0001)
print(res)
    
    
    