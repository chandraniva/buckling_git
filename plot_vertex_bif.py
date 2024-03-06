import numpy as np
import matplotlib.pyplot as plt


Ds_i = 0.0
Ds_f = 0.3

params = [[21,3],[23,3],[59,5],[61,5]]


"""---------------------- Lambda ------------------------"""
i = 0
for p in params:
    N, l0 = p
    
    Ds_abv, lms_abv = np.load("data/vertex_abv_lms_N="+str(N)+"_l0="+str(l0)+".npy")
    Ds, lms = np.load("data/vertex_below_lms_N="+str(N)+"_l0="+str(l0)+".npy")
        
    i_trg = np.where(abs(Ds-Ds_abv[0])==min(abs(Ds - Ds_abv[0])))[0][0]
    
    plt.plot(Ds[:i_trg+1],lms[:i_trg+1],'.-',color='C'+str(i))
    plt.plot(Ds_abv,lms_abv,'.-',label='N='+str(N)+";l0="+str(l0),color='C'+str(i))
    
    i+=1

# plt.xlim(Ds_i,Ds_f)
plt.title("Vertex: Lambda")
plt.legend()
plt.savefig("vertex_lambda_bif.png",dpi=500)
plt.show()

"""---------------------- Amplitude ------------------------"""

i=0
for p in params:
    N, l0 = p
    
    Ds_abv, amp_abv = np.load("data/vertex_abv_amp_N="+str(N)+"_l0="+str(l0)+".npy")
    Ds, amp = np.load("data/vertex_below_amp_N="+str(N)+"_l0="+str(l0)+".npy")
    
    i_trg = np.where(abs(Ds-Ds_abv[0])==min(abs(Ds - Ds_abv[0])))[0][0]
    
    plt.plot(Ds[:i_trg+1],np.abs(amp[:i_trg+1]),'.-',color='C'+str(i))
    plt.plot(Ds_abv,np.abs(amp_abv),'.-',label='N='+str(N)+";l0="+str(l0),color='C'+str(i))
    
    i+=1

# plt.xlim(Ds_i,Ds_f)
plt.title("Vertex: Amplitude")
plt.legend()
plt.savefig("vertex_amp_bif.png",dpi=500)
plt.show()