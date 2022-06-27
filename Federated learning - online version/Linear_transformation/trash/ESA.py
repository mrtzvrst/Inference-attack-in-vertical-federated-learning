import sys
a = 'C:\\Users\\mrtzv\\Desktop\\leetcode\\Data_disclosure_Borzoo\\Federated learning'
if a not in sys.path:
    sys.path.append(a)
    
import numpy as np
import torch, pickle
from My_centralized_codes import ESA, Inf_Att_RG_Half_Zero

torch.set_default_dtype(torch.float64)

"""Extracting the pre-trained model"""
LOSS = 'ESA'
#'Drive_' 
#'Satellite_'
#'Bank_'
#'Robot_'
#'Syn_v1_'
STR = 'v1_'

#'LR_model_Syn_v1.pt'
#'LR_model_bank.pt'
#'LR_model_robot.pt'
#'LR_model_Satellite.pt'
#'LR_model_Sensorless_drive_diagnosis.pt'
f = open('LR_v1.pt', 'rb')
model_data = pickle.load(f)
f.close()

#model_data = torch.load('LR_model_Sensorless_drive_diagnosis.pt')
Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[3], model_data[4]

k0_stp, i_stp, Num_of_Predictions = 1, 1, 100

MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended')
MSE_RG = Inf_Att_RG_Half_Zero(Weights, X, k0_stp, i_stp, Num_of_Predictions, STR)
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC')




