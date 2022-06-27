import sys
a = 'C:\\Users\\mrtzv\\Desktop\\leetcode\\Data_disclosure_Borzoo\\Federated learning'
if a not in sys.path:
    sys.path.append(a)
    
import torch 
import numpy as np
from torch import nn, optim
from My_centralized_codes import random_mini_batches, Read_data, Log_Reg, NeuralNet_Tanh, NeuralNet_Relu, Outlier_detection
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
import pickle

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)

"""Read the data"""
Type = 'LR' 
read = 2
input_dim, output_dim, epochs, batch_size, seed = 5, 2, 1000, 1000, 0
Lambda, learning_rate = 0, 0.05 #Done
if read==1:
    STR = Type+'_v1.pt'    
    X,Y = make_classification(n_samples=50000, n_features=input_dim, n_informative=1, n_redundant=0, n_repeated=0, 
                              n_classes=output_dim, n_clusters_per_class=1, weights=None, 
                              flip_y=0.1, class_sep=1.0, hypercube=True, 
                              shift=1.0, scale=1.0, shuffle=True, random_state=None)
    
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    """Train/test/validation set"""
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=40)
    
    X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
    Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
             
elif read == 2:
    f = open('LR_v1.pt', 'rb')
    model = pickle.load(f)
    f.close()
    
    X, Y = model[1], model[2]
    M = np.random.rand(input_dim,input_dim)
    X = np.matmul(X,M)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size = 0.5, random_state=40)
    
    X_train, X_test, X_valid = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_valid)
    Y_train, Y_test, Y_valid = torch.LongTensor(Y_train), torch.LongTensor(Y_test), torch.LongTensor(Y_valid)
    
"""Train"""
def train(batch_size, X_train, Y_train, seed):
    model.train()
    num_mini_batches = int(X_train.shape[0]/batch_size)
    
    temp = float('inf')
    acc_prev = 0.0
    for i in range(epochs+1):
        train_loss = 0
        
        seed += 1
        minibatches = random_mini_batches(X_train, Y_train, batch_size, seed)
        for MB_X, MB_Y in minibatches:
            
            optimizer.zero_grad()
            Y_out = model.forward(MB_X)
            loss = CEF_loss(Y_out,MB_Y) + Lambda* sum([torch.sum(p**2) for p in model.parameters()])
            loss.backward()
            train_loss += loss.item() 
            optimizer.step()
        
        train_loss/=num_mini_batches
        if i%100==0:
            print('loss after epoch %i: %f'%(i, train_loss))
            if train_loss > temp:
                optimizer.param_groups[0]['lr'] /= 1.5
                #print('learning rate changed:%f'%optimizer.param_groups[0]['lr'])
            temp = train_loss
        
        if i%40 == 0:
            Y_out1 = model(X_valid)
            Y_out2 = torch.argmax(nn.functional.softmax(Y_out1, dim=1), dim=1)
            acc_next = 0
            for i in range(len(Y_valid)):
                if Y_valid[i]==Y_out2[i]:
                    acc_next+=1
            acc_next/=Y_valid.shape[0]
            if acc_prev < acc_next:
                acc_prev = acc_next
                Final_model = model
                print('Validated accuracy is: %f'%acc_next)
    return Final_model


"""Test"""
def test(X_test, Y_test, X_train, Y_train):
    Final_model.eval()
    Y_out1 = Final_model(X_test)
    Y_out2 = torch.argmax(nn.functional.softmax(Y_out1, dim=1), dim=1)
    
    Y_out3 = Final_model(X_train)
    Y_out4 = torch.argmax(nn.functional.softmax(Y_out3, dim=1), dim=1)
    
    accuracy = 0
    for i in range(len(Y_test)):
        if Y_test[i]==Y_out2[i]:
            accuracy+=1
            
    accuracy_t = 0
    for i in range(len(Y_train)):
        if Y_train[i]==Y_out4[i]:
            accuracy_t+=1
        
    accuracy/=Y_test.shape[0]
    accuracy_t/=Y_train.shape[0]
    print('Test accuracy is %f'%accuracy)
    print('Train accuracy is %f'%accuracy_t)
            
if Type == 'NN_Relu': model = NeuralNet_Relu(input_dim, output_dim)
elif Type == 'NN_Tanh': model = NeuralNet_Tanh(input_dim, output_dim)
elif Type == 'LR': model = Log_Reg(input_dim, output_dim)
CEF_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
Final_model = train(batch_size, X_train, Y_train, seed)
test(X_test, Y_test, X_train, Y_train)

# torch.save((Final_model, 
#             X_train.detach().numpy(), 
#             Y_train.detach().numpy(), 
#             X_test.detach().numpy(), 
#             Y_test.detach().numpy(), 
#             X_valid.detach().numpy(), 
#             Y_valid.detach().numpy()), STR)
     
PARAM = (Final_model, X, Y,
            X_train.detach().numpy(), 
            Y_train.detach().numpy(), 
            X_test.detach().numpy(), 
            Y_test.detach().numpy(), 
            X_valid.detach().numpy(), 
            Y_valid.detach().numpy(),
            Lambda, learning_rate)

f = open(STR, 'wb')
pickle.dump(PARAM, f)
f.close()




