import torch 
import pickle
from My_centralized_codes import GIA_NN, GIA_LR
from os.path import exists

torch.set_default_dtype(torch.float64)

LOSS = 'MSE' # Bor1, Bor2, KLD1, KLD2, MSE

MSE_tol = 0.5 #bank: 0.4, robot: 0.45, satellite: 0.4
method = '' #NN_Relu_, NN_Tanh_, ''
limit = 'Clamp'# Clamp, NoClamp
init = 'zero' # half, zero, rand
STR = 'Robot_' # Sesorless_drive_, Satellite_, Bank_, Robot_
filename = STR+method+LOSS+'_'+limit+'_'+init+'_init.pckl'

#'LR_model_bank.pt'
#'LR_model_Satellite.pt'
#'LR_model_robot.pt'
#'NN_Relu_model_bank.pt'
#'NN_Relu_model_Satellite.pt'
#'NN_Relu_model_robot.pt'
#'NN_Tanh_model_bank.pt'
#'NN_Tanh_model_Satellite.pt'
#'NN_Tanh_model_robot.pt'
f = open('NN_Tanh_model_robot.pt', 'rb')
model_data = pickle.load(f)
f.close()

learning_rate = 0.01
T = 50000
loss_th = 10**-12
k0_stp, i_stp, Num_of_Predictions = 1, 1, 100
# model_data = torch.load('LR_model_Satellite.pt')

"""Read the data"""
X, Y = torch.tensor(model_data[3]), torch.LongTensor(model_data[4])

if exists(filename):
    f = open(filename, 'rb')
    MSE = pickle.load(f)
    f.close()
    k0_start = MSE[-1][0]+1
else:
    k0_start, i_start, MSE = 0, 0, []

if method[0:2] == 'NN':
    for param in model_data[0].parameters():
        param.requires_grad = False
    MSE = GIA_NN(model_data[0], X, Y, 
                 learning_rate, k0_stp, i_stp, Num_of_Predictions, T, LOSS, loss_th, 
                 filename, init, limit, MSE_tol, k0_start, MSE)
else:
    """Read the data"""
    Weights = list(model_data[0].named_parameters())[0][1].detach().cpu()
    Biases = list(model_data[0].named_parameters())[1][1].detach().cpu()
    
    GIA_LR(Weights, Biases, X, Y, 
           learning_rate, k0_stp, i_stp, Num_of_Predictions, T, LOSS, loss_th, 
           filename, init, limit, MSE_tol, k0_start, MSE)

