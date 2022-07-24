# import sys
# a = '\\Federated learning'
# if a not in sys.path:
#     sys.path.append(a)

import os
import numpy as np
import torch, pickle
os.chdir('..')
from code_repository import Bounds
os.chdir(os.getcwd()+'\\LS_Halfstar_LB_UB')


torch.set_default_dtype(torch.float64)

"""Extracting the pre-trained model"""
LOSS = 'ESA'
#'Drive_' 
#'Satellite_'
#'Bank_'
#'Robot_'
#'Syn_v1_'
STR = 'Satellite_'

#'LR_model_Syn_v1.pt'
#'LR_model_bank.pt'
#'LR_model_robot.pt'
#'LR_model_Satellite.pt'
#'LR_model_Sensorless_drive_diagnosis.pt'
f = open('LR_model_Satellite.pt', 'rb')
model_data = pickle.load(f)
f.close()

Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[3], model_data[4]
Bounds(Weights, X, Y, STR)


