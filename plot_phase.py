import numpy as np
import matplotlib.pyplot as plt


Ec = np.array([10,12,15,18,21,25,29.30,34.99,46.935,69.94,93.131,117.35])
l0c_sq = np.array([8.132,9.942,12.650,15.275,18.006,21.457,25,30,40,60,80,100])

l0i = np.array([8,10,16,22,25,30,40,60,80,100])
Ei = np.array([23.1,29,46.8,64,72.5,87,116,174,232,290])

l0s_sq = np.linspace(8.1,100,100)

Es1 = np.sqrt(l0s_sq)

plt.figure(figsize=(7,5))

plt.plot(l0c_sq,Ec,'o-',color='red',label='bifurcation line')
plt.plot(l0s_sq,1.15*l0s_sq,'--',color='black',label='St. line fit')
plt.plot(l0i,Ei,'o-',color='navy',label=r'Self-intersection at $D=D_{\Delta}$')
plt.plot(l0s_sq,2.9*l0s_sq,'--',color='red',label='St. line fit')

plt.fill_between(l0s_sq,Es1,color='black',zorder=1)
plt.fill_between(l0c_sq,Ec,color='yellow',zorder=0)
plt.fill_between(l0c_sq,Ec,300*np.ones_like(Ec),color='green',zorder=0)
plt.fill_between(l0i,Ei,300*np.ones_like(Ei), color='grey',zorder=1)
# plt.xlim(8,10)
# plt.ylim(9,12)
plt.ylabel(r"$\Xi$",fontsize=20)
plt.xlabel(r"$(l_0)^2$",fontsize=17)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.legend(fontsize=14)
plt.savefig("phase_interp.png",dpi=500)
plt.show()