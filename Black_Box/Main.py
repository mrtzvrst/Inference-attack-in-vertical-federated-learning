import numpy as np
import os, torch, pickle
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

"""Extracting the pre-trained model"""
#'Satellite_'
#'Bank_'
#'Robot_'
#'Syn_v1_'
STR = 'Bank_'


#'LR_model_Syn_v1.pt'
#'LR_model_bank.pt'
#'LR_model_robot.pt'
#'LR_model_Satellite.pt'
os.chdir('..')
f = open('LR_model_bank.pt', 'rb')
model_data = pickle.load(f)
f.close()
from code_repository import ESA, Inf_Att_RG_Half_Zero
os.chdir(os.getcwd()+'\\Black_Box')


#model_data = torch.load('LR_model_Sensorless_drive_diagnosis.pt')
Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[3], model_data[4]
C = Weights.shape[0]


pas_ind = 14 #11, 12, 14, 17
X_pas = X[:,pas_ind]
X_act = X[:,[j for j in range(19) if j != pas_ind]]
Wpas = Weights[:, pas_ind]
Wact = Weights[:, [j for j in range(19) if j != pas_ind]]


b0 = np.matmul(X_act, Wact.T) # we extract the bias related to the active (we dont have this in reality but we need to make sure wb>0)
# print(b0.shape)
Befor_SM = np.matmul(X, Weights.T)+Biases.reshape(1,-1)
# print(Befor_SM.shape)
b1 = Befor_SM-b0 # we update the bias again (for only choosing those wb>0)
Adversary_observation = b1[:,0]-b1[:,1]


w = Wpas[0]-Wpas[1]
b = Biases[0]-Biases[1]


# print(w*b>0)
if w*b>0:
    MSE = []
    for i in range(2,100, 5):
        
        ite = 0
        MSE_val = 0
        ite_val = 1000
        while ite<ite_val:
            ind3 = np.random.randint(0, X.shape[0], i)
            vM = np.max(Adversary_observation[ind3])
            vm = np.min(Adversary_observation[ind3])
            alpha = (Adversary_observation[ind3]-vM)/(Adversary_observation[ind3]-vm)
            # ind1 = np.where(abs(alpha)==float('inf'))[0]
            # alpha[ind1] = 0.
            MSE_val += np.sum((1/(1-alpha)-X_pas[ind3])**2)/len(alpha)
            ite+=1
        
        MSE.append([MSE_val/ite_val, i])
        
    
    t1 = [i[0] for i in MSE]
    t2 = [i[1] for i in MSE]
    plt.plot(t2,t1,'-o')
    plt.show()
    plt.grid()
    plt.ylabel('MSE', fontsize=15)
    plt.xlabel('N', fontsize=15)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    