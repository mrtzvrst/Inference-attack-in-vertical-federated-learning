import numpy as np
import math
import pickle
import scipy as sc
import cvxpy as cp

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



"""Reading the data and splitting it into test and training (return numpy array)"""
# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
def Read_data(filename):
    if filename=='Sensorless_drive_diagnosis.txt':
        with open(filename) as f:
            lines = f.readlines()
        X, Y = [], []
        for line in lines:
            temp = list(map(float, line.split())) 
            X += [temp[0:48]]
            Y += [int(temp[48]-1)]  
        return np.array(X), np.array(Y)
    
    elif filename=='Sat_train.txt': #NB. There are no examples with class 6 in this dataset.
        with open(filename) as f:
            lines = f.readlines()
        f.close()
        Xtr, Ytr = [], []
        for line in lines:
            temp = list(map(float, line.split())) 
            Xtr += [temp[0:36]]
            if temp[36]==7: Ytr += [int(temp[36]-2)]  
            else: Ytr += [int(temp[36]-1)]  
        
        with open('Sat_test.txt') as f:
            lines = f.readlines()
        f.close()
        Xts, Yts = [], []
        for line in lines:
            temp = list(map(float, line.split())) 
            Xts += [temp[0:36]]
            if temp[36]==7: Yts += [int(temp[36]-2)]  
            else: Yts += [int(temp[36]-1)]   
            
        return np.array(Xtr), np.array(Xts), np.array(Ytr), np.array(Yts)
    
    elif filename == 'bank-additional-full.csv':
        with open(filename) as f:
            lines = f.readlines()
        X, Y = [], []
        for line in lines:
            temp = list(line.replace("\"","").strip().split(';')) 
            X += [temp]
        
        X = X[1:]
        for i in range(len(X)):
            X[i][20] = 0 if X[i][20]=='no' else 1
            
        for i in range(20):
            if X[0][i].replace('.','',1).isdigit():
                for j in range(len(X)):
                    X[j][i] = float(X[j][i])
            else:
                t0 = set()
                for j in range(len(X)):
                    t0.add(X[j][i])        
                
                Dict0={}
                for j in t0:
                    for l in range(len(X)):
                        if X[l][i]==j and j not in Dict0:
                            Dict0[j] = [X[l][20], 1]
                        elif X[l][i]==j:
                            Dict0[j]= [Dict0[j][0]+X[l][20], Dict0[j][1]+1]
                    Dict0[j][0]/=Dict0[j][1]
                
                for j in range(len(X)):
                    X[j][i] = Dict0[X[j][i]][0]
        X = np.array(X)
        return X[:,np.delete(np.arange(20), 11)], X[:,20]
    
    elif filename == 'sensor_readings_24_data.txt':
        with open(filename) as f:
            lines = f.readlines()
        f.close()
        X, Y = [], []
        for line in lines:
            temp = line.strip().split(',')
            X += [list(map(float, temp[0:24]))]
            Y += [temp[24]]  
            
        SET, Dict, ind = set(Y), {}, 0
        for i in SET:
            Dict[i] = ind
            ind+=1
        
        for i in range(len(Y)):
            Y[i] = Dict[Y[i]]
        return np.array(X), np.array(Y)
        
    
    
def Outlier_detection(X, Y, contam_factor=0.1):
        # identify outliers in the training dataset
        iso = IsolationForest(contamination=contam_factor)
        indices = iso.fit_predict(X)
        # select all rows that are not outliers
        mask = indices != -1
        return X[mask, :], Y[mask]
    


"""Define the Logistic Regression model (softmax is automatically defined in the loss function)"""
class Log_Reg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Log_Reg, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        #self.linear = torch.nn.Linear(input_dim, output_dim, bias = False)
    def forward(self, x):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.linear(x)

class NeuralNet_Relu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet_Relu, self).__init__()
        self.layers = nn.Sequential(
          # First upsampling
          nn.Linear(input_dim, 8, bias=True),
          nn.BatchNorm1d(8),
          nn.ReLU(),
          # Second upsampling
          nn.Linear(8, 8, bias=True),
          nn.BatchNorm1d(8),
          nn.ReLU(),
          # Third upsampling
          nn.Linear(8, output_dim, bias=True),
        )
    def forward(self, x):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.layers(x)
    
class NeuralNet_Tanh(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet_Tanh, self).__init__()
        self.layers = nn.Sequential(
          # First upsampling
          nn.Linear(input_dim, 8, bias=True),
          nn.BatchNorm1d(8),
          nn.Tanh(),
          # Second upsampling
          nn.Linear(8, 8, bias=True),
          nn.BatchNorm1d(8),
          nn.Tanh(),
          # Third upsampling
          nn.Linear(8, output_dim, bias=True),
        )
    def forward(self, x):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.layers(x)

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


"""Random guess attack"""
def Inf_Att_RG_Half_Zero(Weights, X, k0_stp, i_stp, Num_of_Predictions, STR):
    Num_of_Features, Num_of_samples = Weights.shape[1], X.shape[0]
    t0, MSE_RG, MSE_Half, MSE_Zero, k0 = int(0.9*Num_of_Features), [], [], [], 0
    while k0 < Num_of_Features:
        print('k0: ', k0)
        i = 0
        while i < t0:
            missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)#np.arange(47,47-i-1,-1) #np.arange(12,i+1+12)  #sample(Feature_index, i+1) #np.arange(0,i+1) 
            ind, MSE_temp0, MSE_temp1, MSE_temp2 = 0, 0, 0, 0
            while ind< Num_of_Predictions:
                index = np.random.randint(0, Num_of_samples)
                ind+=1
                MSE_temp0 += np.sum((np.random.rand(i+1)-X[index][missing_features])**2)/(i+1)
                MSE_temp1 += np.sum((np.ones(i+1)/2-X[index][missing_features])**2)/(i+1)
                MSE_temp2 += np.sum(X[index][missing_features]**2)/(i+1)
            MSE_RG += [[k0, i, MSE_temp0/Num_of_Predictions]]
            MSE_Half += [[k0, i, MSE_temp1/Num_of_Predictions]]
            MSE_Zero += [[k0, i, MSE_temp2/Num_of_Predictions]]
            i+=i_stp
        k0+=k0_stp  
        
    f = open(STR+'RG.pckl', 'wb')
    pickle.dump(MSE_RG, f)
    f.close()
    
    f = open(STR+'Half.pckl', 'wb')
    pickle.dump(MSE_Half, f)
    f.close()
    
    f = open(STR+'Zero.pckl', 'wb')
    pickle.dump(MSE_Zero, f)
    f.close()
        
    return MSE_RG


"""Algorithm 1: GIA attack (No Clamp)"""
def GIA_LR(Weights, Biases, X, Y,
           learning_rate, k0_stp, i_stp, Num_of_Predictions, T, LOSS, loss_th,
           filename, init, limit, MSE_tol, k0_start, MSE):
    # https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/3
    class Clamp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input.clamp(min=0, max=1)
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()
    clamp_class = Clamp()
    
    Num_of_Features = Weights.shape[1]
    Num_of_samples = X.shape[0]
    
    t0 = int(0.9*Num_of_Features)
    
    k0 = k0_start
    while k0 < Num_of_Features:
        i = 0
        while i < t0:
                    
            """Number of the total feature"""
            Missing_Features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)
            ind, MSE_temp = 0, 0
            while ind < Num_of_Predictions:
                index = np.random.randint(0, Num_of_samples) #np.random.choice(Prediction_samples, size=1)[0] #np.random.randint(0, Num_of_Samples)  Data index
                x_act = X[index, [i0 for i0 in range(Num_of_Features) if i0 not in Missing_Features]] # Reading one example from the data for prediction
                # x_pas = torch.zeros(len(Missing_Features), requires_grad=True) # Initialize the passive features to zero
                if init=='zero': x_pas = torch.zeros(len(Missing_Features)) # Initialize the passive features to zero
                elif init=='half': x_pas = torch.ones(len(Missing_Features))/2
                elif init=='rand': x_pas = torch.rand(len(Missing_Features))
                x_pas.requires_grad=True
                W_act = Weights[:, [i0 for i0 in range(Num_of_Features) if i0 not in Missing_Features]]
                W_pas = Weights[:, Missing_Features]
                
                optimizer = optim.Adam([x_pas], lr = learning_rate, betas=(0.9, 0.999), eps=1e-08)
                c = F.softmax(torch.matmul(Weights, X[index]) + Biases, dim=0)
                
                temp_loss0, temp_loss1, temp_loss2, stp = float('inf'), float('inf'), float('inf'), 0
                while temp_loss0>loss_th:
                    if limit == 'Clamp':
                        clamp = clamp_class.apply
                        xp_t = clamp(x_pas)
                        c_hat = F.softmax(torch.matmul(W_act, x_act) + torch.matmul(W_pas, xp_t) + Biases, dim=0)
                    else:
                        c_hat = F.softmax(torch.matmul(W_act, x_act) + torch.matmul(W_pas, x_pas) + Biases, dim=0)
                    loss = Loss_function(LOSS, c, c_hat)
                    
                    """collecting the best solution during the optimization"""
                    if loss.item()<temp_loss1:
                        temp_loss1 = loss.item()
                        xp = x_pas
                    
                    """modifying the learning rate if loss increases every 100 step"""
                    if stp%200==0:
                        if loss.item()<temp_loss2: 
                            temp_loss2 = loss.item()
                        else: 
                            optimizer.param_groups[0]['lr']/=2
                            if optimizer.param_groups[0]['lr']<10**-5:
                                break
                            #print(optimizer.param_groups[0]['lr'])            
                        
                    temp_loss0 = loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step() 
                    stp += 1
                    
                if limit=='Clamp': xp = torch.clamp(xp, min=0, max=1)
                MSE_res = torch.sum((xp-X[index][Missing_Features])**2).detach().numpy()/(i+1)
                
                if MSE_res<MSE_tol and len(Missing_Features)>(len(c)-1): MSE_temp += MSE_res
                elif MSE_res<10**-3 and len(Missing_Features)<len(c): MSE_temp += MSE_res
                else: ind-=1
                                
                ind+=1
                #print([ind, index, MSE_res])
            
            MSE += [[k0, i, MSE_temp/Num_of_Predictions]]
            print([k0, i, np.round(MSE_temp/Num_of_Predictions, decimals=5)])   

            i+=i_stp
        k0+=k0_stp
        
        f = open(filename, 'wb')
        pickle.dump(MSE, f)
        f.close()
        
    return MSE


"""Algorithm 1: GIA attack (No Clamp)"""
def GIA_NN(model, X, Y,
           learning_rate, k0_stp, i_stp, Num_of_Predictions, T, LOSS, loss_th,
           filename, init, limit, MSE_tol, k0_start, MSE):
    
    # https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/3
    class Clamp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input.clamp(min=0, max=1)
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()
    clamp_class = Clamp()
    
    Num_of_Features = X.shape[1]
    Num_of_samples = X.shape[0]
    
    t0 = int(0.9*Num_of_Features)
 
    
    k0 = k0_start
    while k0 < Num_of_Features:
        i = 0
        while i < t0:
                    
            """Number of the total feature"""
            Missing_Features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)
            ind, MSE_temp = 0, 0
            while ind < Num_of_Predictions:
                index = np.random.randint(0, Num_of_samples) #np.random.choice(Prediction_samples, size=1)[0] #np.random.randint(0, Num_of_Samples)  Data index
                x_samp = torch.clone(X[index]) # Reading one example from the data for prediction
                x_samp[Missing_Features] = 0
                mask = torch.zeros(Num_of_Features)
                mask[Missing_Features] = 1.
                
                if init=='zero': xp = torch.zeros(Num_of_Features) # Initialize the passive features to zero
                elif init=='half': xp = torch.ones(Num_of_Features)/2
                elif init=='rand': xp = torch.rand(Num_of_Features)
                xp.requires_grad=True
                
                optimizer = optim.Adam([xp], lr = learning_rate, betas=(0.9, 0.999), eps=1e-08)
                c = F.softmax(model(X[index].reshape(1,Num_of_Features)), dim=1)
                
                temp_loss0, temp_loss1, temp_loss2, stp = float('inf'), float('inf'), float('inf'), 0
                while temp_loss0>loss_th:
                    if limit == 'Clamp':
                        clamp = clamp_class.apply
                        xp_t = clamp(xp)
                        c_hat = F.softmax(model((mask*xp_t+x_samp).reshape(1,Num_of_Features)), dim=1)
                    else:
                        c_hat = F.softmax(model((mask*xp+x_samp).reshape(1,Num_of_Features)), dim=1)
                    loss = Loss_function(LOSS, c, c_hat)
                    
                    """collecting the best solution during the optimization"""
                    if loss.item()<temp_loss1:
                        temp_loss1 = loss.item()
                        x_pas = xp[Missing_Features]
                        
                    """modifying the learning rate if loss increases every 100 step"""
                    if stp%200==0:
                        if loss.item()<temp_loss2: 
                            temp_loss2 = loss.item()
                        else: 
                            optimizer.param_groups[0]['lr']/=2
                            if optimizer.param_groups[0]['lr']<10**-5:
                                break
                            #print(optimizer.param_groups[0]['lr'])            
                        
                    temp_loss0 = loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step() 
                    stp += 1
        
                if limit=='Clamp': x_pas = torch.clamp(x_pas, min=0, max=1)
                MSE_res = torch.sum((x_pas-X[index][Missing_Features])**2).detach().numpy()/(i+1)
                
                if MSE_res<MSE_tol and len(Missing_Features)>(len(c[0])-1): MSE_temp += MSE_res
                elif ('Relu' in filename) and MSE_res<MSE_tol: MSE_temp += MSE_res
                elif ('Relu' not in filename) and MSE_res<10**-3 and len(Missing_Features)<len(c[0]): MSE_temp += MSE_res
                else: ind-=1
                
                #print(torch.round(x_pas,decimals=2),'\n',torch.round(X[index][Missing_Features],decimals=2))
                ind+=1
                
            MSE += [[k0, i, MSE_temp/Num_of_Predictions]]
            print([k0, i, np.round(MSE_temp/Num_of_Predictions, decimals=5)])   

            i+=i_stp
        k0+=k0_stp
        
        f = open(filename, 'wb')
        pickle.dump(MSE, f)
        f.close()
        
    return MSE



# def ESA_Accuracy(Weights, Biases, missing_features, X, Truncate, Version):
    
#     Num_of_Features = Weights.shape[1]
#     Num_of_classes = Weights.shape[0]
   
#     res = []
#     i = len(missing_features)-1
#     for l in range(X.shape[0]):
        
#         z = np.matmul(Weights, X[l,:]) + Biases
#         #z = np.matmul(Weights, X[index])
#         v = sc.special.softmax(z, axis=0)
        
#         Wpas = Weights[:, missing_features]
#         Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
#         X_act = X[l, [j for j in range(Num_of_Features) if j not in missing_features]]
            
#         if 0<= i < Num_of_classes-1:
#             W, B = np.concatenate((Wpas, Wact), axis=1), Biases[0:-1]-Biases[1:]
#             #W = np.concatenate((Wpas, Wact), axis=1)
#             W, A = W[0:-1, :]-W[1:, :], np.log(v[0:-1])-np.log(v[1:])
#             X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1]-B[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
#             #X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
        
#         else:
#             #A = np.log(v)-np.matmul(Wact, X_act)
#             A = np.log(v)-np.matmul(Wact, X_act)-Biases
#             Wpas, A = Wpas[0:-1,:]-Wpas[1:,:], A[0:-1]-A[1:]
#             if Version == 'Extended':
#                 m0 = np.matmul(np.linalg.pinv(Wpas), A)[:,None]
#                 m1 = np.identity(i+1)-np.matmul(np.linalg.pinv(Wpas), Wpas)
#                 m2 = np.matmul(np.matmul(m1, np.linalg.pinv(m1)), np.ones((i+1,1))-1/2-m0)
#                 X_pas = (m0 + m2).flatten()
                                        
#             else:                        
#                 X_pas = np.matmul(np.linalg.pinv(Wpas), A)
        
#         """Truncation"""
#         if Truncate=='yes':
#             X_pas[X_pas<0], X_pas[X_pas>1] = 0, 1
        
#         res += [X_pas]
#     return res
        
        
#     # i = len(missing_features)-1
#     # Num_of_Features = Weights.shape[1]
#     # Num_of_classes = Weights.shape[0]
    
#     # Wpas = Weights[:, missing_features]
#     # Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
     
#     # W, B = np.concatenate((Wpas, Wact), axis=1), Biases[0:-1]-Biases[1:]
#     # W = W[0:-1, :]-W[1:, :]
    
#     # Wpas = Wpas[0:-1,:]-Wpas[1:,:]
#     # m1 = np.identity(i+1)-np.matmul(np.linalg.pinv(Wpas), Wpas)
    
#     # x_hat = []
#     # for x in X:
#     #     z = np.matmul(Weights, x) + Biases
#     #     v = sc.special.softmax(z, axis=0)
#     #     X_act = x[[j for j in range(Num_of_Features) if j not in missing_features]]
            
#     #     if 0<= i < Num_of_classes-1:
#     #         A = np.log(v[0:-1])-np.log(v[1:])
#     #         X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1]-B[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
            
#     #     else:
#     #         A = np.log(v)-np.matmul(Wact, X_act)-Biases
#     #         A = A[0:-1]-A[1:]
#     #         if Version == 'Extended':
#     #             m0 = np.matmul(np.linalg.pinv(Wpas), A)[:,None]
#     #             m2 = np.matmul(np.matmul(m1, np.linalg.pinv(m1)), np.ones((i+1,1))-1/2-m0)
#     #             X_pas = (m0 + m2).flatten()
                                        
#     #         else:                        
#     #             X_pas = np.matmul(np.linalg.pinv(Wpas), A)
        
#     #     """Truncation"""
#     #     if Truncate=='yes':
#     #         X_pas[X_pas<0], X_pas[X_pas>1] = 0, 1
        
#     #     x_hat += [X_pas]
        
#     # return x_hat



# def ESA_Accuracy1(Weights, Biases, missing_features, X, Truncate, Version):
    
  
        
#     i = len(missing_features)-1
#     Num_of_Features = Weights.shape[1]
#     Num_of_classes = Weights.shape[0]
    
#     Wpas = Weights[:, missing_features]
#     Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
     
#     W, B = np.concatenate((Wpas, Wact), axis=1), Biases[0:-1]-Biases[1:]
#     W = W[0:-1, :]-W[1:, :]
    
#     Wpas = Wpas[0:-1,:]-Wpas[1:,:]
#     m1 = np.identity(i+1)-np.matmul(np.linalg.pinv(Wpas), Wpas)
    
#     x_hat = []
#     for x in X:
#         z = np.matmul(Weights, x) + Biases
#         v = sc.special.softmax(z, axis=0)
#         X_act = x[[j for j in range(Num_of_Features) if j not in missing_features]]
            
#         if 0<= i < Num_of_classes-1:
#             A = np.log(v[0:-1])-np.log(v[1:])
#             X_pas = np.matmul( np.linalg.inv(W[0:i+1, 0:i+1]), (A[0:i+1]-B[0:i+1])-np.matmul(W[0:i+1, i+1:],X_act))
            
#         else:
#             A = np.log(v)-np.matmul(Wact, X_act)-Biases
#             A = A[0:-1]-A[1:]
#             if Version == 'Extended':
#                 m0 = np.matmul(np.linalg.pinv(Wpas), A)[:,None]
#                 m2 = np.matmul(np.matmul(m1, np.linalg.pinv(m1)), np.ones((i+1,1))-1/2-m0)
#                 X_pas = (m0 + m2).flatten()
                                        
#             else:                        
#                 X_pas = np.matmul(np.linalg.pinv(Wpas), A)
        
#         """Truncation"""
#         if Truncate=='yes':
#             X_pas[X_pas<0], X_pas[X_pas>1] = 0, 1
        
#         x_hat += [X_pas]
        
#     return x_hat