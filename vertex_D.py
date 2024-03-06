import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib.patches import Polygon
from scipy import interpolate

l0 = 2.2
N = 15
n = int((N + 1)/2)
 
Di = 0.0
Df = 0.5
D_steps = 20
D_steps_abv = 10

def iso_trap_vertices(x, y, l0, phi, psi, lambd):
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

def energy(x):
    phis = x[:-1]
    lambd = x[-1]
    e = np.sum(l0**2 / lambd / np.cos(phis[1:-1]) + lambd) + \
            (l0**2 / lambd / np.cos(phis[0]) + lambd)/2 + \
            (l0**2 / lambd / np.cos(phis[-1]) + lambd)/2
    return e

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
    
    return xs[-1] - (N-1)/(2*l0) * (1 - D)



# phi_init = np.zeros(n)
# lambd_init = l0
# phi_init_flat = phi_init.flatten()
# x0 = np.concatenate([phi_init_flat, [lambd_init]])


z = np.pi**2/4/(N/2)**2
D_star = 1-np.sqrt(1-z)

Ds = np.linspace(Di,Df,D_steps)
Lambds = np.zeros_like(Ds)
amps = np.zeros_like(Ds)
trg_cond = np.zeros_like(Ds)

ii = 0

for D in Ds:
    if D<D_star:
        phi_init = np.zeros(n)
        lambd_init = l0
        phi_init_flat = phi_init.flatten()
        x0 = np.concatenate([phi_init_flat, [lambd_init]])
    elif D>D_star and D<D_star+0.005:
        sigma = np.arange(n)/n
        eps = np.sqrt(1 - 1*(1-D))
        psi_init = eps*2/(1-np.pi**2/4/(N/2)**2)* np.sin(np.pi*sigma)
        dpsi_init = eps*2/(1-np.pi**2/4/(N/2)**2)* np.cos(np.pi*sigma) * np.pi
        phi_init = dpsi_init/N
        
        lambd_init = l0
        phi_init_flat = phi_init.flatten()
        x0 = np.concatenate([phi_init_flat, [lambd_init]])
        
    
    b = (-10,10)
    bounds = [b for _ in range(n)]+[(0.0*l0,4*l0)]
    
    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'eq', 'fun': constraint2, 'args': (D,)}
    con = [con1, con2]
    
    sol = optimize.minimize(energy, x0, constraints=con,
                            bounds = bounds)
    x_opt = sol.x
    phis = x_opt[:-1]
    lambd = x_opt[-1]

    
    x0 = x_opt 
    
    Lambds[ii] = lambd/l0
    
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
        
        
    amps[ii] = ys[-1]
    trg_cond[ii] = np.abs(np.sin(2*phis[-1])) - 2/lambd**2
    # plt.plot(psis,'.-',label='psi')
    # plt.plot(phis,'.-',label='phi')
    # plt.legend()
    # plt.show()
    
    if D<0.1:
        phi_abv_init = phis
        lambd_abv_init = lambd


    for i in range(n):
        vertices = np.transpose(iso_trap_vertices(xs[i],ys[i],l0,phis[i],psis[i],lambd))
        plt.scatter(xs[i], ys[i],s = 3,color='red') 
        plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))
        xc = xs[-1]
        reflected_vertices = np.transpose(iso_trap_vertices(-xs[i]+2*xc,
                                        ys[i],l0,phis[i],-psis[i],lambd))
        plt.scatter(-xs[i]+2*xc, ys[i],s=3,color='red') 
        plt.gca().add_patch(Polygon(reflected_vertices, closed=True, fill=None, edgecolor='b'))
        
    plt.title("Below D_trg: D="+str(D))
    # plt.grid(True)
    plt.axis([-1,N/l0,-N/2/l0,N/2/l0])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("figures/vertex/cells/below_cells_N="+str(N)+
    #             "_l0="+str(l0)+"_D="+str(D)+".png",dpi=500)
    plt.show()
    
    ii+=1


plt.plot(Ds,Lambds,'.-')
plt.axvline(x=D_star,linestyle='--',c='red')
# plt.ylim(0.9,1.1)
plt.title("Lambda: N="+str(N)+"; l0="+str(l0))
plt.xlabel("D")
# plt.savefig("vertex_lambd_N="+str(N)+"_l0="+str(l0)+".png",dpi = 500)
plt.show()


plt.plot(Ds,amps,'.-')
plt.axvline(x=D_star,linestyle='--',c='red')
# plt.ylim(0.9,1.1)
plt.title("Amplitude: N="+str(N)+"; l0="+str(l0))
plt.xlabel("D")
# plt.savefig("vertex_amplitude_N="+str(N)+"_l0="+str(l0)+".png",dpi = 500)
plt.show()


plt.plot(Ds,trg_cond,'.-')

f = interpolate.UnivariateSpline(Ds, trg_cond, s=0)
yToFind = 0
yreduced = np.array(trg_cond) - yToFind
freduced = interpolate.UnivariateSpline(Ds, yreduced, s=0)
D_trg = freduced.roots()[0]
plt.axvline(x=D_trg,linestyle='--',c='red',linewidth=3)
plt.xlabel("D")
plt.ylabel("trg_cond")
# plt.savefig("trg_cond.png",dpi=500)
plt.show()



print("-------------above D_trg------------")

def energy_abv(x):
    Phis = x[:-1]
    lambd = x[-1]
    phi_max = np.arcsin(2/lambd**2)/2
    phis = phi_max*np.sin(Phis)
    
    e = np.sum(l0**2 / lambd / np.cos(phis[1:-1]) + lambd) + \
            (l0**2 / lambd / np.cos(phis[0]) + lambd)/2 + \
            (l0**2 / lambd / np.cos(phis[-1]) + lambd)/2 
    return e

def constraint1_abv(x):
    Phis = x[:-1]
    lambd = x[-1]
    phi_max = np.arcsin(2/lambd**2)/2
    phis = phi_max*np.sin(Phis)
    psis = np.zeros_like(phis)
    
    for i in range(1,n):
        psis[i] = psis[i-1] + phis[i] + phis[i-1]
    
    return psis[-1]

def constraint2_abv(x, D):
    Phis = x[:-1]
    lambd = x[-1]
    phi_max = np.arcsin(2/lambd**2)/2
    phis = phi_max*np.sin(Phis)
    
    psis = np.zeros_like(phis)
    
    xs = np.zeros_like(phis)
    
    for i in range(1,n):
        psis[i] = psis[i-1] + phis[i] + phis[i-1]
    
 
    for i in range(1,n):
        xs[i] = xs[i-1] + (np.cos(psis[i])/np.cos(phis[i]) + 
                   np.cos(psis[i-1])/np.cos(phis[i-1])) /lambd/2
    
    return xs[-1] - (N-1)/(2*l0) * (1 - D)




phi_max_init = np.arcsin(2/lambd_abv_init**2)/2
Phi_abv_init = np.arcsin(phi_abv_init/phi_max_init)
Phi_abv_init_flat = Phi_abv_init.flatten()

x0_abv = np.concatenate([Phi_abv_init_flat, [lambd_abv_init]])

Ds_abv = np.linspace(D_trg,Df,D_steps_abv)
Lambds_abv = np.zeros_like(Ds_abv)
amps_abv = np.zeros_like(Ds_abv)

ii = 0

for D in Ds_abv:
    b = (-10,10)
    
    bounds = [b for _ in range(n)]+[(0.05*l0,5*l0)]
    
    con1 = {'type': 'eq', 'fun': constraint1_abv}
    con2 = {'type': 'eq', 'fun': constraint2_abv, 'args': (D,)}
    con = [con1, con2]
    
    sol = optimize.minimize(energy_abv, x0_abv, constraints=con,
                            bounds = bounds)
    x_opt = sol.x
    Phis = x_opt[:-1]
    lambd = x_opt[-1]
    phis = np.arcsin(2/lambd**2)/2 * np.sin(Phis)
    
    x0_abv = x_opt
    
    Lambds_abv[ii] = lambd/l0
    
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
        
        
    amps_abv[ii] = ys[-1]
    # plt.plot(psis,'.-',label='psi')
    # plt.plot(phis,'.-',label='phi')
    # plt.legend()
    # plt.show()
    # if D == Dx_abv:
    #     for i in range(n):
    #         vertices = np.transpose(iso_trap_vertices(xs[i],ys[i],l0,phis[i],psis[i],lambd))
    #         plt.scatter(xs[i], ys[i],s = 3,color='red') 
    #         plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))
    #         xc = xs[-1]
    #         reflected_vertices = np.transpose(iso_trap_vertices(-xs[i]+2*xc,
    #                                         ys[i],l0,phis[i],-psis[i],lambd))
    #         plt.scatter(-xs[i]+2*xc, ys[i],s=3,color='red') 
    #         plt.gca().add_patch(Polygon(reflected_vertices, closed=True, fill=None, edgecolor='b'))
        
    #     plt.title("Above D_trg: D="+str(D))
    #     plt.grid(True)
    #     plt.axis([-1,N/l0,-N/2/l0,N/2/l0])
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.savefig("cells_N="+str(N)+"_l0="+str(l0)+"_D="+str(D)+".png",dpi=500)
    #     plt.show()

    for i in range(n):
        vertices = np.transpose(iso_trap_vertices(xs[i],ys[i],l0,phis[i],psis[i],lambd))
        plt.scatter(xs[i], ys[i],s = 3,color='red') 
        plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))
        xc = xs[-1]
        reflected_vertices = np.transpose(iso_trap_vertices(-xs[i]+2*xc,
                                        ys[i],l0,phis[i],-psis[i],lambd))
        plt.scatter(-xs[i]+2*xc, ys[i],s=3,color='red') 
        plt.gca().add_patch(Polygon(reflected_vertices, closed=True, fill=None, edgecolor='b'))
        
    plt.title("Above D_trg: D="+str(D))
    # plt.grid(True)
    plt.axis([-1,N/l0,-N/2/l0,N/2/l0])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("figures/vertex/cells/abv_cells_N="+str(N)+"_l0="+
    #             str(l0)+"_D="+str(D)+".png",dpi=500)
    plt.show()
    
    ii+=1

plt.plot(Ds,Lambds,'.-')
plt.axvline(x=D_star,linestyle='--',c='red')
plt.plot(Ds_abv,Lambds_abv,'.-')
plt.axvline(x=D_trg,linestyle='--',c='red')
# plt.ylim(0.9,1.1)
plt.title("Lambda: N="+str(N)+"; l0="+str(l0))
plt.xlabel("D")
# plt.savefig("vertex_lambd_N="+str(N)+"_l0="+str(l0)+".png",dpi = 500)
plt.show()

plt.plot(Ds,amps,'.-')
plt.axvline(x=D_star,linestyle='--',c='red')
plt.plot(Ds_abv,amps_abv,'.-')
plt.axvline(x=D_trg,linestyle='--',c='red')
plt.title("Amplitude: N="+str(N)+"; l0="+str(l0))
plt.xlabel("D")
# plt.savefig("vertex_amplitude_N="+str(N)+"_l0="+str(l0)+".png",dpi = 500)
plt.show()



np.save("data/vertex_abv_lms_N="+str(N)+"_l0="+str(l0)+
        ".npy",np.vstack((Ds_abv,Lambds_abv)))
np.save("data/vertex_abv_amp_N="+str(N)+"_l0="+str(l0)+
        ".npy",np.vstack((Ds_abv,amps_abv)))
np.save("data/vertex_below_lms_N="+str(N)+"_l0="+str(l0)+
        ".npy",np.vstack((Ds,Lambds)))
np.save("data/vertex_below_amp_N="+str(N)+"_l0="+str(l0)+
        ".npy",np.vstack((Ds,amps)))
# np.save("data/vertex_intsct_E="+str(E)+"_l0^2="+str(round(l0**2*1000)/1000)+
#         "_tol="+str(tol)+".npy",np.array(D_intsct))