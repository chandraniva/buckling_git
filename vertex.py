import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib.patches import Polygon

l0 = 1
N = 21
n = int((N + 1) / 2)
D = 0.3

def iso_trap_vertices(x, y, h, phi, psi,lambd):
    # Calculate half-width of the trapezoid's top base
    w_top = (1/h-lambd*l0*np.sin(phi))/2

    # Calculate half-width of the trapezoid's bottom base
    w_bottom = (1/h+lambd*l0*np.sin(phi))/2

    # Calculate x-coordinates of the vertices of the top base
    x_left_top = x - w_top
    x_right_top = x + w_top

    # Calculate y-coordinates of the vertices of the top and bottom bases
    y_top = y + h/2
    y_bottom = y - h/2

    # Rotate the vertices around the center
    rotation_matrix = np.array([
        [np.cos(psi), np.sin(psi)],
        [-np.sin(psi), np.cos(psi)]
    ])
    vertices = np.array([
        [x_left_top, y_top],   # Top left
        [x_right_top, y_top],  # Top right
        [x + w_bottom, y_bottom],  # Bottom right
        [x - w_bottom, y_bottom]   # Bottom left
    ])
    rotated_vertices = np.dot(vertices - [x, y], rotation_matrix) + [x, y]
    

    return rotated_vertices

def energy(x):
    phis = x[:-1]
    lambd = x[-1]
    return np.sum(l0**2 / lambd / np.cos(phis) + lambd)

def constraint1(x):
    phis = x[:-1]
    lambd = x[-1]
    psi = 0
    phi_prev = 0
    for i in range(1, n):
        phi = phis[i]
        phi_prev = phis[i-1]
        psi += phi + phi_prev
    return psi

def constraint2(x, D):
    phis = x[:-1]
    lambd = x[-1]
    xs = np.zeros_like(phis)
    phi_prev = phis[0]
    psi_prev = 0
    psi = 0
    x_val = 0
    
    for i in range(1, n):
        phi = phis[i]
        phi_prev = phis[i-1]
        x_val += (np.cos(psi)/np.cos(phi) + 
                   np.cos(psi_prev)/np.cos(phi_prev)) /lambd/2
        xs[i] = x_val
        psi_prev = psi
        psi += phi + phi_prev
    
        
    return x_val - N / (2 * l0) * (1 - D)


phi_init = np.ones(n)*0
lambd_init = 1
phi_init_flat = phi_init.flatten()
x0 = np.concatenate([phi_init_flat, [lambd_init]])

b = (-2,2)
bounds = [b for _ in range(n)]+[(0.2,2)]

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
phi_prev = phis[0]
psi_prev = 0
psi = 0
x_val = 0
y_val = 0

polygons = []

for i in range(1, n):
    phi = phis[i]
    phi_prev = phis[i-1]
    
    x_val += (np.cos(psi)/np.cos(phi) + 
               np.cos(psi_prev)/np.cos(phi_prev)) / lambd /2
    xs[i] = x_val
    
    y_val += (np.sin(psi) / np.cos(phi) + 
                    np.sin(psi_prev)/np.cos(phi_prev)) / lambd /2
    ys[i] = y_val
    
    psi += phi + phi_prev
    psis[i] = psi
    
    
    
    
print("lambda =",lambd)
plt.plot(psis,'.-')
plt.show()



# plt.plot(xs,ys,'-')
for i in range(n):
    h = lambd*l0*np.cos(phis[i])
    vertices = iso_trap_vertices(xs[i],ys[i],h,phis[i],psis[i],lambd)
    plt.scatter(xs[i], ys[i],color='red') 
    plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))

plt.axis('equal')
plt.show()
plt.show()






# for i, vertices in enumerate(polygons, start=1):
#     plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))