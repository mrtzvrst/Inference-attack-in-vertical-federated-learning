import sys
a = 'C:\\Users\\mrtzv\\Desktop\\leetcode\\Data_disclosure_Borzoo\\Federated learning'
if a not in sys.path:
    sys.path.append(a)

import pickle
import torch 
import numpy as np

from My_centralized_codes import ESA_Accuracy, Read_data
# https://www.datatechnotes.com/2020/09/sgd-classification-example-with-sgdclassifier-in-python.html

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.set_default_dtype(torch.float64)


LOSS = 'ESA'
#'Drive_' 
#'Satellite_'
#'Bank_'
#'Robot_'
STR = 'Bank_'
Lngth = 50
"""Read the data"""
Read = 3 #1: Drive, 2: Satellite, 3: Bank, 4: Robot


if Read == 2:#satellite
    X_train, X_test, Y_train, Y_test = Read_data('Sat_train.txt')
    """Normalization"""
    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
    X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
    input_dim, output_dim = 36, 6
    
elif Read == 3:#bank
    X, Y= Read_data('bank-additional-full.csv')
    """Normalization"""
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    """Train/test/validation set"""
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
    input_dim, output_dim = 19, 2
    
elif Read == 4:#robot
    X, Y = Read_data('sensor_readings_24_data.txt')
    """Normalization"""
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    """Train/test/validation set"""
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
    input_dim, output_dim = 24, 4


Num_of_Features = input_dim
f = open('LR_model_bank.pt', 'rb')
model_data = pickle.load(f)
f.close()
    
Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu())
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu())

res1, res2, res3, res4, res5, res6 = [], [], [], [], [], []
t0 = int(0.9*Num_of_Features)
k0 = 0
while k0 < Num_of_Features:
    print(k0)
    i = 0
    while i < t0:
        missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)
        
        X_tr = X_train[:,missing_features]
        sgdc = SGDClassifier(loss='log', max_iter=2000, tol=0.0001)# 
        sgdc.fit(X_tr,Y_train)
        score = sgdc.score(X_tr, Y_train)
        #print("Training score: ", score) 
        
        X_ts0 = X_test[np.random.randint(X_test.shape[0], size=Lngth),:]
        X_ts1 = X_ts0[:,missing_features]
        Y_ts1 = sgdc.predict(X_ts1)
        X_hat1 = np.array(ESA_Accuracy(Weights, Biases, missing_features, X_ts0, 'yes', 'Non_extension'))
        Y_pr1 = sgdc.predict(X_hat1)
        res1 += [[k0, i, 1-np.sum(Y_pr1!=Y_ts1)/Lngth]]
        
        X_ts0 = X_test[np.random.randint(X_test.shape[0], size=Lngth),:]
        X_ts1 = X_ts0[:,missing_features]
        Y_ts1 = sgdc.predict(X_ts1)
        X_hat2 = np.array(ESA_Accuracy(Weights, Biases, missing_features, X_ts0, 'yes', 'Extended'))
        Y_pr2 = sgdc.predict(X_hat2)
        res2 += [[k0, i, 1-np.sum(Y_pr2!=Y_ts1)/Lngth]]
        
        X_ts0 = X_test[np.random.randint(X_test.shape[0], size=Lngth),:]
        X_ts1 = X_ts0[:,missing_features]
        Y_ts1 = sgdc.predict(X_ts1)        
        X_hat3 = np.array(ESA_Accuracy(Weights, Biases, missing_features, X_ts0, 'no', 'Non_extension'))
        Y_pr3 = sgdc.predict(X_hat3)
        res3 += [[k0, i, 1-np.sum(Y_pr3!=Y_ts1)/Lngth]]
        
        X_hat4 = np.ones(X_ts0[:,missing_features].shape)/2
        Y_pr4 = sgdc.predict(X_hat4)
        res4 += [[k0, i, 1-np.sum(Y_pr4!=Y_ts1)/Lngth]]
        
        X_hat5 = np.zeros(X_ts0[:,missing_features].shape)
        Y_pr5 = sgdc.predict(X_hat5)
        res5 += [[k0, i, 1-np.sum(Y_pr5!=Y_ts1)/Lngth]]
        
        X_hat6 = np.random.rand(X_ts0[:,missing_features].shape[0],X_ts0[:,missing_features].shape[1])
        Y_pr6 = sgdc.predict(X_hat6)
        res6 += [[k0, i, 1-np.sum(Y_pr6!=Y_ts1)/Lngth]]
        #print([k0, i, 1-np.sum(Y_pr3!=Y_ts1)/Lngth])
        
        i+=1
    k0+=1
        
        
f = open('Accuracy_'+STR+LOSS+'_Clamp.pckl', 'wb')
pickle.dump(res1, f)
f.close()

f = open('Accuracy_'+STR+LOSS+'_Clamp_Extended.pckl', 'wb')
pickle.dump(res2, f)
f.close()

f = open('Accuracy_'+STR+LOSS+'_NoClamp.pckl', 'wb')
pickle.dump(res3, f)
f.close()

f = open('Accuracy_'+STR+'half.pckl', 'wb')
pickle.dump(res4, f)
f.close()

f = open('Accuracy_'+STR+'zero.pckl', 'wb')
pickle.dump(res5, f)
f.close()

f = open('Accuracy_'+STR+'random.pckl', 'wb')
pickle.dump(res6, f)
f.close()