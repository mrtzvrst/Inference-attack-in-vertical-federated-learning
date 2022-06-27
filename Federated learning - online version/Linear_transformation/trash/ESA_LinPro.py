from scipy.optimize import linprog
import scipy as sc
import numpy as np
import torch, pickle
from My_centralized_codes import ESA, Inf_Att_RG_Half_Zero, ESA_Extended

torch.set_default_dtype(torch.float64)

#'LR_model_bank.pt'
#'LR_model_bank_reg_0p001.pt'
#'LR_model_bank_reg_0p1.pt'
#'LR_model_Satellite.pt'
#'LR_model_Sensorless_drive_diagnosis.pt'
f = open('LR_model_bank.pt', 'rb')
model_data = pickle.load(f)
f.close()

#model_data = torch.load('LR_model_Sensorless_drive_diagnosis.pt')
Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[3], model_data[4]
Num_of_Predictions = 1

Num_of_Features = Weights.shape[1]
Num_of_classes = Weights.shape[0]
Num_of_samples = X.shape[0]

t0 = int(0.9*Num_of_Features)
MSE = []
k0, i = 0, 9
missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)#np.arange(47,47-i-1,-1) #np.arange(12,i+1+12)  #sample(Feature_index, i+1) #np.arange(0,i+1) 
MSE_temp = 0
index = np.random.randint(0, Num_of_samples)
z = np.matmul(Weights, X[index]) + Biases
#z = np.matmul(Weights, X[index])
v = sc.special.softmax(z, axis=0)

Wpas = Weights[:, missing_features]
Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
X_act = X[index][[j for j in range(Num_of_Features) if j not in missing_features]]






A = np.log(v)-np.matmul(Wact, X_act)-Biases
#A = np.log(v)-np.matmul(Wact, X_act)
Wpas, A = Wpas[0:-1,:]-Wpas[1:,:], A[0:-1]-A[1:]
s0 = np.linalg.pinv(Wpas)
C = np.matmul(s0, A)[:,None]
w, v = np.linalg.eig(np.matmul(s0, Wpas))
w[np.abs(w)<0.5]=0.
v0 = v[:,w == 0]

com_ind = np.where(np.sum(np.iscomplex(v0), axis=0)!=0)[0]
# for l in range(v0.shape[1]):
#     if np.sum(np.imag(v0[:,l]))
    

# https://realpython.com/linear-programming-python/
bb = np.real(v0[:,com_ind])
b0 = np.diag(np.matmul(np.transpose(bb), bb))
tt = {}
for l in range(len(b0)):
    tt[b0[l]] = l

ttt = np.concatenate((np.real(v0[:, [ind for ind in range(v0.shape[1]) if ind not in com_ind]]), bb[:, [ind for ind in tt.values()]]*2), axis=1)

    
obj = np.ones(ttt.shape[1])
lhs_ineq = np.concatenate((ttt, -ttt), axis=0)
rhs_ineq  = np.concatenate((np.ones((ttt.shape[0],1))-C, C), axis=0)

temp = (0,1)
bnd = []
for l in range(ttt.shape[1]):
    bnd+=[temp]
    
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
              bounds=bnd,method="revised simplex")





# MSE_temp = np.sum((X_pas-X[index][missing_features])**2)/(i+1)

   
    
