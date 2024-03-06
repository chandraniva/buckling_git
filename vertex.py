import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib.patches import Polygon

l0 = 2
N = 41
n = int((N + 1) / 2)
D = 0.4


def iso_trap_vertices(x, y, l0, phi, psi, lambd):
    Lambd = lambd/l0
    
    #height of the trapezoid
    h = lambd*np.cos(phi)
    
    # Calculate half-width of the trapezoid's top base
    w_top = (1/h-lambd*np.sin(phi))/2

    # Calculate half-width of the trapezoid's bottom base
    w_bottom = (1/h+lambd*np.sin(phi))/2

    # Calculate x-coordinates of the vertices of the top base
    x_left_top = x - w_top
    x_right_top = x + w_top
    x_left_bottom = x - w_bottom
    x_right_bottom = x + w_bottom

    # Calculate y-coordinates of the vertices of the top and bottom bases
    y_top = y + h/2
    y_bottom = y - h/2

    # Rotate the vertices around the center
    rotation_matrix = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    vertices = np.transpose(np.array([
        [x_left_top, y_top],   # Top left
        [x_right_top, y_top],  # Top right
        [x_right_bottom, y_bottom],  # Bottom right
        [x_left_bottom, y_bottom]   # Bottom left
    ]))
    rotated_vertices = np.dot(rotation_matrix,vertices - [[x],[y]]) + [[x],[y]]
    

    return rotated_vertices
    # return vertices

def energy(x):
    phis = x[:-1]
    lambd = x[-1]
    return np.sum(l0**2 / lambd / np.cos(phis) + lambd)

def constraint1(x):
    phis = x[:-1]
    lambd = x[-1]
    
    psis = np.zeros_like(phis)
    
    for i in range(1,n):
        psis[i] = psis[i-1] + phis[i] + phis[i-1]
    
    return psis[-1]

def constraint2(x, D):
    phis = x[:-1]
    lambd = x[-1]
    
    psis = np.zeros_like(phis)
    
    xs = np.zeros_like(phis)
    
    for i in range(1,n):
        psis[i] = psis[i-1] + phis[i] + phis[i-1]
    

    for i in range(1,n):
        xs[i] = xs[i-1] + (np.cos(psis[i])/np.cos(phis[i]) + 
                   np.cos(psis[i-1])/np.cos(phis[i-1])) /lambd/2
    
        
    return xs[-1] - N / (2* l0) * (1 - D)


phi_init = np.ones(n)*0
lambd_init = 1
phi_init_flat = phi_init.flatten()
x0 = np.concatenate([phi_init_flat, [lambd_init]])

b = (-1,1)
bounds = [b for _ in range(n)]+[(0.1,2)]

con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2, 'args': (D,)}
con = [con1, con2]

sol = optimize.minimize(energy, x0, constraints=con,
                        bounds = bounds)
x_opt = sol.x
phis = x_opt[:-1]
lambd = x_opt[-1]


psis = np.zeros_like(phis)
xs = np.zeros_like(phis)
ys = np.zeros_like(phis)

for i in range(1,n):
    psis[i] = psis[i-1] + phis[i] + phis[i-1]

for i in range(1,n):
    xs[i] = xs[i-1] + (np.cos(psis[i])/np.cos(phis[i]) + 
               np.cos(psis[i-1])/np.cos(phis[i-1])) /lambd/2
    ys[i] = ys[i-1] + (np.sin(psis[i])/np.cos(phis[i]) + 
               np.sin(psis[i-1])/np.cos(phis[i-1])) /lambd/2
    
    
    
    
print("lambda =",lambd)
plt.plot(psis,'.-',label='psi')
plt.plot(phis,'.-',label='phi')
plt.legend()
plt.show()



# plt.plot(xs,ys,'-')
for i in range(n):
    vertices = np.transpose(iso_trap_vertices(xs[i],ys[i],l0,phis[i],psis[i],lambd))
    plt.scatter(xs[i], ys[i],color='red') 
    plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))
    xc = xs[-1]
    reflected_vertices = np.transpose(iso_trap_vertices(-xs[i]+2*xc,
                                    ys[i],l0,phis[i],-psis[i],lambd))
    plt.scatter(-xs[i]+2*xc, ys[i],color='red') 
    plt.gca().add_patch(Polygon(reflected_vertices, closed=True, fill=None, edgecolor='b'))
    
# plt.xlim(-1,N/2/l0)
# plt.ylim(-5,5)
plt.axis('equal')
plt.grid(True)
# plt.savefig("cells_N="+str(N)+"_l0="+str(l0)+"_D="+str(D)+".png",dpi=500)
plt.show()
