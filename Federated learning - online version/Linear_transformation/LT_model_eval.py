import torch 
import numpy as np
from torch import nn, optim
from LT_codes import random_mini_batches, Log_Reg, test, train, ESA, ESA_transformed
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
import pickle
from os.path import exists

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)

input_dim, output_dim, epochs, batch_size, seed = 5, 2, 200, 1000, 0
Lambda, learning_rate = 0, 0.05
k0_stp, i_stp, Num_of_Predictions, accuracy = 1, 1, 1000, 0.94

"""First we train the data without any transformation"""
Model_NT_name = 'LR_Model_No_Trans.pckl'    
if not exists(Model_NT_name):
    X,Y = make_classification(n_samples=50000, n_features=input_dim, n_informative=1, n_redundant=0, n_repeated=0, 
                              n_classes=output_dim, n_clusters_per_class=1, weights=None, 
                              flip_y=0.1, class_sep=1.0, hypercube=True, 
                              shift=1.0, scale=1.0, shuffle=True, random_state=None)
    
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    #Train/test/validation set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=40)
    
    X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
    Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
    
    #Training the data
    model_NT = Log_Reg(input_dim, output_dim) # Model with No Transformation
    CEF_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_NT.parameters(), lr=learning_rate)
    Final_model_NT = train(model_NT, epochs, optimizer, CEF_loss, Lambda, batch_size, X_train, Y_train, X_valid, Y_valid, seed)
    _ = test(Final_model_NT, X_test, Y_test, X_train, Y_train)
    
    #Saving the model and params
    PARAM = (Final_model_NT, X, Y, X_train.detach().numpy(), Y_train.detach().numpy(), X_test.detach().numpy(), Y_test.detach().numpy(), X_valid.detach().numpy(), Y_valid.detach().numpy(), Lambda, learning_rate)
    
    f = open(Model_NT_name, 'wb')
    pickle.dump(PARAM, f)
    f.close()

LOSS = 'ESA'

"""Results for the model without Linear transformation"""
STR = 'Bef_Trans_'
f = open(Model_NT_name, 'rb')
model_data = pickle.load(f)
f.close()

Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[1], model_data[2]

MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC')
MSE = ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS')


"""Results for the model with Linear transformation"""
STR = 'Aft_Trans_'
MSE = ESA_transformed(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp.pckl', STR, 'yes', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name)
MSE = ESA_transformed(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_Clamp_Extended.pckl', STR, 'yes', 'Extended', epochs, batch_size, accuracy, Model_NT_name)
MSE = ESA_transformed(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_NoClamp.pckl', STR, 'no', 'Non_extension', epochs, batch_size, accuracy, Model_NT_name)
MSE = ESA_transformed(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_0centre.pckl', STR, 'no', '0centre', epochs, batch_size, accuracy, Model_NT_name)
MSE = ESA_transformed(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_RCC.pckl', STR, 'no', 'RCC', epochs, batch_size, accuracy, Model_NT_name)
MSE = ESA_transformed(k0_stp, i_stp, Num_of_Predictions, STR+LOSS+'_CLS.pckl', STR, 'no', 'CLS', epochs, batch_size, accuracy, Model_NT_name)
