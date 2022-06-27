import numpy as np
import math
import pickle
import scipy as sc
import cvxpy as cp
from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

import torch
from torch import nn, optim
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)

"""Randomly choosing mini-batched of data"""
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



"""Define the Logistic Regression model (softmax is automatically defined in the loss function)"""
class Log_Reg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Log_Reg, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        #self.linear = torch.nn.Linear(input_dim, output_dim, bias = False)
    def forward(self, x):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.linear(x)



"""Loss function (MSE and KLD)"""
def Loss_function(flag, c, c_hat):
    if flag == 'MSE':
        return nn.MSELoss(reduction='mean')(c,c_hat)
    
    elif flag == 'KLD1':
        return torch.sum(c*(torch.log(c)-torch.log(c_hat)))
    
    elif flag == 'KLD2':
        return torch.sum(c_hat*(torch.log(c_hat)-torch.log(c)))
    
    elif flag == 'Bor1':
        return torch.sum((torch.log(c_hat)-torch.log(c))**2)
    
    elif flag == 'Bor2':
        return torch.sum((c/c_hat-1)**2)
    

def My_plot(d_tot, filename):
    f = open(filename, 'rb')
    MSE = pickle.load(f)
    f.close()
            
    Dict = {}
    for i in range(len(MSE)):
        if MSE[i][1] not in Dict: Dict[MSE[i][1]] = [MSE[i][2], 1]
        else: Dict[MSE[i][1]] = [MSE[i][2]+Dict[MSE[i][1]][0], Dict[MSE[i][1]][1]+1]
    
    MSE_plt = np.zeros(len(Dict))
    t1 = MSE_plt.copy()
    ind = 0
    for i in Dict:
        MSE_plt[ind] = Dict[i][0] / Dict[i][1]
        t1[ind] = (i+1)/d_tot*100
        ind+=1
    return t1, MSE_plt
    

"""ESA attack"""
def ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, filename, STR, Truncate, Version):
    
    
    Num_of_Features = Weights.shape[1]
    Num_of_classes = Weights.shape[0]
    Num_of_samples = X.shape[0]
    
    t0 = int(0.9*Num_of_Features)
    MSE = []
    k0 = 0
    while k0 < Num_of_Features:
        i = 0
        while i < t0:
            
            missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)#np.arange(47,47-i-1,-1) #np.arange(12,i+1+12)  #sample(Feature_index, i+1) #np.arange(0,i+1) 
            ind, MSE_temp = 0, 0
            while ind< Num_of_Predictions:
                
                index = np.random.randint(0, Num_of_samples)
                
                z = np.matmul(Weights, X[index]) + Biases
                #z = np.matmul(Weights, X[index])
                v = sc.special.softmax(z, axis=0)
                
                Wpas = Weights[:, missing_features]
                Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
                X_act = X[index][[j for j in range(Num_of_Features) if j not in missing_features]]
                    
                if 0<= i < Num_of_classes-1:
                    W, B = np.concatenate((Wpas, Wact), axis=1), Biases[0:-1]-Biases[1:]
                    #W = np.concatenate((Wpas, Wact), axis=1)
                    W, A = W[0:-1, :]-W[1:, :], np.log(v[0:-1])-np.log(v[1:])
                    X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1]-B[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
                    #X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
                
                else:
                    #A = np.log(v)-np.matmul(Wact, X_act)
                    A = np.log(v)-np.matmul(Wact, X_act)-Biases
                    Wpas, A = Wpas[0:-1,:]-Wpas[1:,:], A[0:-1]-A[1:]
                    if Version == 'Extended':
                        m0 = np.matmul(np.linalg.pinv(Wpas), A)[:,None]
                        m1 = np.identity(i+1)-np.matmul(np.linalg.pinv(Wpas), Wpas)
                        m2 = np.matmul(np.matmul(m1, np.linalg.pinv(m1)), np.ones((i+1,1))-1/2-m0)
                        X_pas = (m0 + m2).flatten()
                        
                    elif Version == '0centre':
                        low = -np.matmul(np.linalg.pinv(Wpas), A)
                        upp = 1+low
                        Wpas_null = sc.linalg.null_space(Wpas)
                        Wpas_null_sum = np.sum(Wpas_null, axis=0)
                        n = Wpas_null_sum.shape[0]
                        
                        Alpha = cp.Variable(n)                        
                        objective = cp.Minimize( cp.sum_squares(Alpha) - Wpas_null_sum@Alpha )
                        constraints = [low <= Wpas_null@Alpha, Wpas_null@Alpha <= upp]
                        prob = cp.Problem(objective, constraints)
                        prob.solve(max_iter=100000)
                        X_pas = -low+np.matmul(Wpas_null,Alpha.value)
                    
                    elif Version == 'RCC':
                        Wpas_null = sc.linalg.null_space(Wpas)
                        Wpas_null_sq = Wpas_null**2

                        Q = np.matmul(np.linalg.pinv(Wpas), A).reshape(-1,1)
                        G = (Q-1/2)*Wpas_null#every row is gi for i=1,...,n
                        Qn = Q*(1-Q)

                        n = Wpas_null.shape[0]
                        Alpha = cp.Variable(n)
                        
                        term1 = Alpha@G
                        term2 = Wpas_null.T@cp.diag(Alpha)@Wpas_null
                        
                        temp, flag = 0, True
                        while flag:
                            objective = cp.Minimize( cp.matrix_frac(term1, term2) + Qn.T @ Alpha )
                            constraints = [Alpha>=0+temp , Alpha@Wpas_null_sq>=(1-temp)]
                            prob = cp.Problem(objective, constraints)
                            prob.solve(max_iters=100000)
                            if sum(Alpha.value>=0)==i+1: #and sum((Alpha@Wpas_null_sq).value>=1)==Wpas_null.shape[1]:
                                flag = False
                            else:
                                temp+=10**-5
                                
                        X_pas = Q.T - np.matmul(np.matmul(Wpas_null, np.linalg.inv(term2.value)), term1.value)
                        
                    elif Version == 'CLS':
                        n = len(missing_features)
                        Alpha = cp.Variable(n)
                        
                        objective = cp.Minimize( cp.sum_squares(Wpas @ Alpha - A) )
                        constraints = [Alpha>=0 , Alpha<=1]
                        prob = cp.Problem(objective, constraints)
                        prob.solve(max_iter=30000)
                        X_pas = Alpha.value
                        
                        
                        
                    else:                        
                        X_pas = np.matmul(np.linalg.pinv(Wpas), A)
                
                """Truncation"""
                if Truncate=='yes':
                    X_pas[X_pas<0], X_pas[X_pas>1] = 0, 1
                
                MSE_temp += np.sum((X_pas-X[index][missing_features])**2)/(i+1)
                ind+=1
                #print(np.round(X_pas,2),'\n',np.round(X[index][missing_features],2))
            MSE += [[k0, i, MSE_temp/Num_of_Predictions]] 
            print(k0, i, np.round(MSE_temp/Num_of_Predictions, decimals=5))
            i+=i_stp
        k0+=k0_stp      
        
    f = open(filename, 'wb')
    pickle.dump(MSE, f)
    f.close()
         
    return MSE


"""ESA attack"""
def ESA_transformed(k0_stp, i_stp, Num_of_Predictions, filename, STR, Truncate, Version, epochs, batch_size, accuracy, Model_NT_name):
    
    f = open(Model_NT_name, 'rb')
    model_data = pickle.load(f)
    f.close()
    X1 = model_data[1]
    Num_of_samples = X1.shape[0]
    Num_of_Features = X1.shape[1]
    Lambda, learning_rate, Num_of_classes = model_data[9], model_data[10], len(list(model_data[0].named_parameters())[1][1])
    
    t0 = int(0.9*Num_of_Features)
    MSE = []
    
    k0 = 0
    while k0 < Num_of_Features:
        i = 1 # length of features should be more than one
        while i < t0:
            
            missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)
            fn = 'Model_'+str(k0)+'_'+str(i)+'.pckl'
            if exists(fn):
                f = open(fn, 'rb')
                model_data = pickle.load(f)
                f.close()
                
                Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
                Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
                X, Y = model_data[1], model_data[2]
                
            else:
                f = open(Model_NT_name, 'rb')
                model = pickle.load(f)
                f.close()
                
                X, Y = model[1], model[2]
                v1 = np.linalg.eig(np.matmul(X[:,missing_features].T, X[:,missing_features]))[1][:,0].reshape(-1,1)
                M1 = 10*np.matmul(v1, np.ones(v1.shape).T)+np.identity(v1.shape[0])
                
                X[:,missing_features] = np.matmul(X[:,missing_features],M1)
                X[:,missing_features] = (X[:,missing_features] - np.min(X[:,missing_features], axis=0)) / (np.max(X[:,missing_features], axis=0) - np.min(X[:,missing_features], axis=0))
                
                
                
                model1 = Log_Reg(Num_of_Features, Num_of_classes)
                CEF_loss1 = nn.CrossEntropyLoss()
                optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
                acc=0
                while acc<accuracy:
                    x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size = 0.2, random_state=np.random.randint(0, 100, size=1)[0])
                    x_ts, x_vl, y_ts, y_vl = train_test_split(x_ts, y_ts, test_size = 0.5, random_state=np.random.randint(0, 100, size=1)[0])
                    
                    x_tr, x_ts, x_vl = torch.Tensor(x_tr), torch.Tensor(x_ts), torch.Tensor(x_vl)
                    y_tr, y_ts, y_vl = torch.LongTensor(y_tr), torch.LongTensor(y_ts), torch.LongTensor(y_vl)
                    
                    Final_model1 = train(model1, epochs, optimizer1, CEF_loss1, Lambda, batch_size, x_tr, y_tr, x_vl, y_vl, np.random.randint(0, 1000, size=1)[0])
                    acc = test(Final_model1, x_ts, y_ts, x_tr, y_tr)
                    
                PARAM = (Final_model1, X, Y, x_tr.detach().numpy(), y_tr.detach().numpy(), x_ts.detach().numpy(), y_ts.detach().numpy(), x_vl.detach().numpy(), y_vl.detach().numpy(), Lambda, learning_rate)
                Weights = np.array(list(PARAM[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
                Biases = np.array(list(PARAM[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
                
                f = open(fn, 'wb')
                pickle.dump(PARAM, f)
                f.close()
            
            
        
            ind, MSE_temp = 0, 0
            while ind< Num_of_Predictions:
                
                index = np.random.randint(0, Num_of_samples)
                
                z = np.matmul(Weights, X[index]) + Biases
                #z = np.matmul(Weights, X[index])
                v = sc.special.softmax(z, axis=0)
                
                Wpas = Weights[:, missing_features]
                Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
                X_act = X[index][[j for j in range(Num_of_Features) if j not in missing_features]]
                    
                if 0<= i < Num_of_classes-1:
                    W, B = np.concatenate((Wpas, Wact), axis=1), Biases[0:-1]-Biases[1:]
                    #W = np.concatenate((Wpas, Wact), axis=1)
                    W, A = W[0:-1, :]-W[1:, :], np.log(v[0:-1])-np.log(v[1:])
                    X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1]-B[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
                    #X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
                
                else:
                    #A = np.log(v)-np.matmul(Wact, X_act)
                    A = np.log(v)-np.matmul(Wact, X_act)-Biases
                    Wpas, A = Wpas[0:-1,:]-Wpas[1:,:], A[0:-1]-A[1:]
                    if Version == 'Extended':
                        m0 = np.matmul(np.linalg.pinv(Wpas), A)[:,None]
                        m1 = np.identity(i+1)-np.matmul(np.linalg.pinv(Wpas), Wpas)
                        m2 = np.matmul(np.matmul(m1, np.linalg.pinv(m1)), np.ones((i+1,1))-1/2-m0)
                        X_pas = (m0 + m2).flatten()
                        
                    elif Version == '0centre':
                        low = -np.matmul(np.linalg.pinv(Wpas), A)
                        upp = 1+low
                        Wpas_null = sc.linalg.null_space(Wpas)
                        Wpas_null_sum = np.sum(Wpas_null, axis=0)
                        n = Wpas_null_sum.shape[0]
                        
                        Alpha = cp.Variable(n)                        
                        objective = cp.Minimize( cp.sum_squares(Alpha) - Wpas_null_sum@Alpha )
                        constraints = [low <= Wpas_null@Alpha, Wpas_null@Alpha <= upp]
                        prob = cp.Problem(objective, constraints)
                        prob.solve(max_iter=100000)
                        X_pas = -low+np.matmul(Wpas_null,Alpha.value)
                    
                    elif Version == 'RCC':
                        Wpas_null = sc.linalg.null_space(Wpas)
                        Wpas_null_sq = Wpas_null**2

                        Q = np.matmul(np.linalg.pinv(Wpas), A).reshape(-1,1)
                        G = (Q-1/2)*Wpas_null#every row is gi for i=1,...,n
                        Qn = Q*(1-Q)

                        n = Wpas_null.shape[0]
                        Alpha = cp.Variable(n)
                        
                        term1 = Alpha@G
                        term2 = Wpas_null.T@cp.diag(Alpha)@Wpas_null
                        
                        temp, flag = 0, True
                        while flag:
                            objective = cp.Minimize( cp.matrix_frac(term1, term2) + Qn.T @ Alpha )
                            constraints = [Alpha>=0+temp , Alpha@Wpas_null_sq>=(1-temp)]
                            prob = cp.Problem(objective, constraints)
                            prob.solve(max_iters=100000)
                            if sum(Alpha.value>=0)==i+1: #and sum((Alpha@Wpas_null_sq).value>=1)==Wpas_null.shape[1]:
                                flag = False
                            else:
                                temp+=10**-5
                                
                        X_pas = Q.T - np.matmul(np.matmul(Wpas_null, np.linalg.inv(term2.value)), term1.value)
                        
                    elif Version == 'CLS':
                        n = len(missing_features)
                        Alpha = cp.Variable(n)
                        
                        objective = cp.Minimize( cp.sum_squares(Wpas @ Alpha - A) )
                        constraints = [Alpha>=0 , Alpha<=1]
                        prob = cp.Problem(objective, constraints)
                        prob.solve(max_iter=30000)
                        X_pas = Alpha.value
                        
                        
                        
                    else:                        
                        X_pas = np.matmul(np.linalg.pinv(Wpas), A)
                
                """Truncation"""
                if Truncate=='yes':
                    X_pas[X_pas<0], X_pas[X_pas>1] = 0, 1
                
                MSE_temp += np.sum((X_pas-X1[index][missing_features])**2)/(i+1)
                ind+=1
                #print(np.round(X_pas,2),'\n',np.round(X[index][missing_features],2))
            MSE += [[k0, i, MSE_temp/Num_of_Predictions]] 
            print(k0, i, np.round(MSE_temp/Num_of_Predictions, decimals=5))
            i+=i_stp
        k0+=k0_stp      
        
    f = open(filename, 'wb')
    pickle.dump(MSE, f)
    f.close()
         
    return MSE




"""Train"""
def train(model, epochs, optimizer, CEF_loss, Lambda, batch_size, X_train, Y_train, X_valid, Y_valid, seed):
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
def test(Final_model, X_test, Y_test, X_train, Y_train):
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
    return accuracy