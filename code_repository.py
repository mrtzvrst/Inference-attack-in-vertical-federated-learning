import math, pickle, torch
import numpy as np
import scipy as sc
import cvxpy as cp

from sklearn.ensemble import IsolationForest

from torch import nn, optim
from torch.nn import functional as F
from os.path import exists

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
from torchviz import make_dot

torch.set_default_dtype(torch.float64)

"""
List of finctions:

random_mini_batches 
Read_data               (to read the data from real datasets)
Outlier_detection   
Log_Reg                 (VFL)
NeuralNet_Relu          (VFL)
NeuralNet_Tanh          (VFL)
Loss_function 
My_plot 
Bounds                  (This is for LS and Half*)
ESA                     (LS, Clamped_LS, RCC1, RCC2, CLS)
Inf_Att_RG_Half_Zero    (Zero, Half, Random Guess performance)
GIA_LR                  (GIA approach in Logistic regression)
GIA_NN
train_VFL_NN
test_VFL_NN
"""



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
    def __init__(self, da, dp, output_dim):
        super(NeuralNet_Relu, self).__init__()
        self.layers_act = nn.Sequential(
          # First upsampling
          nn.Linear(da, min(6,da), bias=True),
          # nn.BatchNorm1d(8),
          nn.ReLU(),
          # Second upsampling
          nn.Linear(min(6,da), 8, bias=True),
          # nn.BatchNorm1d(8),
          nn.ReLU(),
          # Third upsampling
          nn.Linear(8, output_dim, bias=True),
        )
        self.layers_pas = nn.Sequential(
          # First upsampling
          nn.Linear(dp, min(6,dp), bias=True),
          # nn.BatchNorm1d(8),
          nn.ReLU(),
          # Second upsampling
          nn.Linear(min(6,dp), 8, bias=True),
          # nn.BatchNorm1d(8),
          nn.ReLU(),
          # Third upsampling
          nn.Linear(8, output_dim, bias=True),
        )
    def forward(self, x1, x2):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.layers_act(x1)+self.layers_pas(x2)
    
  
    
class NeuralNet_Tanh(nn.Module):
    def __init__(self, da, dp, output_dim):
        super(NeuralNet_Tanh, self).__init__()
        # da, dp = 20,16
        # L1 = 8
        # L2 = (da+dp)/L1
        # L3 = max(da, dp)//L2
        # L4 = L1 - L3
        # print(L3, L4)
        
        self.layers_act = nn.Sequential(
          # First upsampling
          nn.Linear(da, min(output_dim-1,da), bias=True),
          # nn.BatchNorm1d(8),
          nn.Tanh(),
          # Second upsampling
          nn.Linear(min(output_dim-1,da), output_dim-1, bias=True),
          # nn.BatchNorm1d(8),
          nn.Tanh(),
          # Third upsampling
          nn.Linear(output_dim-1, output_dim, bias=True),
        )
        self.layers_pas = nn.Sequential(
          # First upsampling
          nn.Linear(dp, min(output_dim-1,dp), bias=True),
          # nn.BatchNorm1d(8),
          nn.Tanh(),
          # Second upsampling
          nn.Linear(min(output_dim-1,dp), output_dim-1, bias=True),
          # nn.BatchNorm1d(8),
          nn.Tanh(),
          # Third upsampling
          nn.Linear(output_dim-1, output_dim, bias=True),
        )
    def forward(self, x1, x2):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.layers_act(x1)+self.layers_pas(x2)
    # def act_forward(self, x1):
    #     return self.layers_act(x1)
    # def pas_forward(self, x2):
    #     return self.layers_pas(x2)
    


"""Loss function (MSE and KLD)"""
def Loss_function(flag, c, c_hat):
    # 'MSE': nn.MSELoss(reduction='mean')(c,c_hat)
    # 'KLD': torch.sum(c*(torch.log(c)-torch.log(c_hat)))
    # 'Bor': torch.sum((c/c_hat-1)**2)
    
    if flag == 'KLD':
        return torch.sum(c_hat*(torch.log(c_hat)-torch.log(c)))
    
    elif flag == 'Bor':
        return torch.sum((torch.log(c_hat)-torch.log(c))**2)
    
    

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
    


"""Lower and upper bounds for LS and Half*"""
def Bounds(Weights, X, Y, STR):
    Num_of_Features = Weights.shape[1]
       
    t0 = int(0.9*Num_of_Features)
    UB11, UB12, UB21, UB22, LB, LS, Half_star = np.zeros((1, t0)), np.zeros((1, t0)), np.zeros((1, t0)), np.zeros((1, t0)), np.zeros((1, t0)), np.zeros((1, t0)), np.zeros((1, t0))
    k0 = 0
    while k0 < Num_of_Features:
        print(k0)
        i = 0
        while i < t0:
            
            missing_features = np.mod(np.arange(k0,i+1+k0), Num_of_Features)#np.arange(47,47-i-1,-1) #np.arange(12,i+1+12)  #sample(Feature_index, i+1) #np.arange(0,i+1) 
            
            dpas = len(missing_features)
            Xn = X[:, missing_features]
            Mean = np.reshape(np.sum(Xn, axis=0)/Xn.shape[0], (-1,1))
            Half = np.reshape(np.ones((1, dpas))/2, (-1,1))
            KZero = np.zeros((dpas, dpas))
            KHalf = np.zeros((dpas, dpas))
            KCova = np.zeros((dpas, dpas))
            for j in range(Xn.shape[0]):
                temp = np.reshape(Xn[j, :], (-1,1))
                KZero += np.matmul(temp, temp.T)
                KHalf += np.matmul(temp-Half, (temp-Half).T)
                KCova += np.matmul(temp-Mean, (temp-Mean).T)
            KZero/=Xn.shape[0]
            KHalf/=Xn.shape[0]
            KCova/=Xn.shape[0]
            
            Wpas = Weights[:, missing_features]
            A = Wpas[0:-1,:]-Wpas[1:,:]
            Aplus = np.linalg.pinv(A)
            
            if dpas!=1:
                Null_A = sc.linalg.null_space(A).shape[1]
                UB11[0,i] += np.trace(KZero)/dpas
                UB12[0,i] += np.sqrt(Null_A)*np.sqrt(np.trace(np.matmul(KZero, KZero)))/dpas
                UB21[0,i] += np.trace(KHalf)/dpas
                UB22[0,i] += np.sqrt(Null_A)*np.sqrt(np.trace(np.matmul(KHalf, KHalf)))/dpas
                LS[0,i] += np.trace( np.matmul( (np.identity(dpas)-np.matmul(Aplus, A)) , KZero) )/dpas
                Half_star[0,i] += np.trace( np.matmul( (np.identity(dpas)-np.matmul(Aplus, A)) , KHalf) )/dpas
                LB[0,i] += np.trace( np.matmul( (np.identity(dpas)-np.matmul(Aplus, A)) , KCova) )/dpas
            i+=1
        k0+=1
    
    f = open(STR+'UB11', 'wb')
    pickle.dump(UB11/Num_of_Features, f)
    f.close()
    
    f = open(STR+'UB12', 'wb')
    pickle.dump(UB12/Num_of_Features, f)
    f.close()
    
    f = open(STR+'UB21', 'wb')
    pickle.dump(UB21/Num_of_Features, f)
    f.close()
    
    f = open(STR+'UB22', 'wb')
    pickle.dump(UB22/Num_of_Features, f)
    f.close()
    
    f = open(STR+'LB', 'wb')
    pickle.dump(LB/Num_of_Features, f)
    f.close()
    
    f = open(STR+'LS', 'wb')
    pickle.dump(LS/Num_of_Features, f)
    f.close()
    
    f = open(STR+'Half_star', 'wb')
    pickle.dump(Half_star/Num_of_Features, f)
    f.close()
    


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
           learning_rate, k0_stp, i_stp, Num_of_Predictions, LOSS, loss_th,
           filename, init, MSE_tol, k0_start, MSE):
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
                    # Clamping the solution
                    clamp = clamp_class.apply
                    xp_t = clamp(x_pas)
                    c_hat = F.softmax(torch.matmul(W_act, x_act) + torch.matmul(W_pas, xp_t) + Biases, dim=0)
                    
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
                
                #Clamping the final solution 
                xp = torch.clamp(xp, min=0, max=1)
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




"""GIA attack (Clamped over NN)"""
def GIA_NN(DATA, k0_start, k0_stp, i_stp, Num_of_Predictions, loss_th, MSE_tol, lr,
            MSE, STR, Type, LOSS, init, filename):
    # https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/3
    class Clamp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input.clamp(min=0, max=1)
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()
    clamp_class = Clamp()
    
    dt = DATA[6] #Number of features 
    X = DATA[2] #X_test for applying GIA on the samples
    
    Num_of_samples = X.shape[0]
    t0 = int(0.9*dt)
    k0 = k0_start
    while k0 < dt:
        i = 0
        while i < t0:
            
            pas_ind = np.mod(np.arange(k0,i+1+k0), dt)
            act_ind = [j for j in range(dt) if j not in pas_ind]
            dp = len(pas_ind)
            da = dt - dp
            
            model_name = STR+Type+'_'+str(k0)+'_'+str(i+1)+'.pckl'
            if not exists(model_name):
                if Type == 'tanh': model = NeuralNet_Tanh(da, dp, DATA[7])
                elif Type == 'relu': model = NeuralNet_Relu(da, dp, DATA[7])
                CEF_loss = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=DATA[8]['lr'])
                model = train_VFL_NN(model, DATA[0], DATA[1], DATA[4], DATA[5], optimizer, CEF_loss, 
                                      DATA[8]['epochs'], DATA[8]['batch_size'], DATA[8]['Lambda'], DATA[8]['seed'],
                                      da, dp, DATA[7], act_ind, pas_ind)
                Test_Acc , _ = test_VFL_NN(model, DATA[2], DATA[3], DATA[0], DATA[1], da, dp, DATA[7], act_ind, pas_ind)
                
                for param in model.parameters():
                    param.requires_grad = False
                    
                f = open(model_name, 'wb')
                pickle.dump((model, Test_Acc), f)
                f.close()
                
            else:
                f = open(model_name, 'rb')
                model = pickle.load(f)[0]
                f.close()
            
            ind, MSE_temp = 0, 0
            while ind < Num_of_Predictions:
                
                index = np.random.randint(0, Num_of_samples) #np.random.choice(Prediction_samples, size=1)[0] #np.random.randint(0, Num_of_Samples)  Data index
                x_sample = torch.clone(X[index]) # Reading one example from the data for prediction
                xa = x_sample[[j for j in range(dt) if j not in pas_ind]]
                xp_real_val = x_sample[pas_ind]
                """Below comment was used in the earlier version to mask the active party"""
                # x_sample[pas_ind] = 0
                # mask = torch.zeros(dt)
                # mask[pas_ind] = 1.
                
                if init=='zero': xp = torch.zeros(dp) # Initialize the passive features to zero
                elif init=='half': xp = torch.ones(dp)/2
                elif init=='rand': xp = torch.rand(dp)
                xp.requires_grad=True
                
                act_added_bias = model.layers_act(xa.reshape(1,da)).detach()
                conf_vector = (model.layers_pas(xp_real_val.reshape(1,dp)) + model.layers_act(xa.reshape(1,da))).detach()
                c = F.softmax(conf_vector, dim=1)
                
                optimizer = optim.Adam([xp], lr = lr, betas=(0.9, 0.999), eps=1e-08)
                temp_loss0, temp_loss1, temp_loss2, stp = float('inf'), float('inf'), float('inf'), 0
                while temp_loss0>loss_th:
                
                    clamp = clamp_class.apply
                    xp_t = clamp(xp)
                    c_hat = F.softmax(act_added_bias + model.layers_pas(xp_t.reshape(1,dp)), dim=1)
                    loss = Loss_function(LOSS, c, c_hat)
                    
                    # yhat = model.layers_pas(xp)
                    # make_dot(c_hat, params=dict(model.layers_pas.named_parameters())).render("passive", format="png", view='True')
                    # make_dot(c_hat, params=dict(model.layers_pas.named_parameters())).view()s
                    
                    """collecting the best solution during the optimization"""
                    if loss.item()<temp_loss1:
                        temp_loss1 = loss.item()
                        x_pas = xp
                        
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
        
                x_pas = torch.clamp(x_pas, min=0, max=1)
                MSE_res = torch.sum((x_pas-X[index][pas_ind])**2).detach().numpy()/(i+1)
                MSE_temp += MSE_res
                """This part should be revisited"""
                # if MSE_res<MSE_tol and len(pas_ind)>(len(c[0])-1): MSE_temp += MSE_res
                # elif ('relu' in filename) and MSE_res<MSE_tol: MSE_temp += MSE_res
                # elif ('relu' not in filename) and MSE_res<10**-3 and len(pas_ind)<len(c[0]): MSE_temp += MSE_res
                # else: ind-=1
                
                #print(torch.round(x_pas,decimals=2),'\n',torch.round(X[index][pas_ind],decimals=2))
                ind+=1
                
            MSE += [[k0, i, MSE_temp/Num_of_Predictions]]
            print([k0, i, np.round(MSE_temp/Num_of_Predictions, decimals=5)])   
            
            i+=i_stp
        k0+=k0_stp
        
        f = open(filename, 'wb')
        pickle.dump(MSE, f)
        f.close()
        
    return MSE


"""Train"""
def train_VFL_NN(model, X_train, Y_train, X_valid, Y_valid, optimizer, CEF_loss, epochs, batch_size, Lambda, seed, da, dp, d_out, act_ind, pas_ind):
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
            Y_out = model.forward(MB_X[:, act_ind], MB_X[:, pas_ind])
            loss = CEF_loss(Y_out,MB_Y) + Lambda* sum([torch.sum(p**2) for p in model.parameters()])
            loss.backward()
            train_loss += loss.item() 
            optimizer.step()
        
        train_loss/=num_mini_batches
        if i%100==0:
            #print('loss after epoch %i: %f'%(i, train_loss))
            if train_loss > temp:
                optimizer.param_groups[0]['lr'] /= 1.5
                #print('learning rate changed:%f'%optimizer.param_groups[0]['lr'])
            temp = train_loss
        
        if i%40 == 0:
            Y_out1 = model(X_valid[:, act_ind], X_valid[:, pas_ind])
            Y_out2 = torch.argmax(nn.functional.softmax(Y_out1, dim=1), dim=1)
            acc_next = 0
            for l in range(len(Y_valid)):
                if Y_valid[l]==Y_out2[l]:
                    acc_next+=1
            acc_next/=Y_valid.shape[0]
            if acc_prev < acc_next:
                acc_prev = acc_next
                Final_model = model
                # print('Validated accuracy is: %f'%acc_next, i)
                
    return Final_model



"""Test"""
def test_VFL_NN(Final_model, X_test, Y_test, X_train, Y_train, da, dp, d_out, act_ind, pas_ind):
    Final_model.eval()
    Y_out1 = Final_model(X_test[:,act_ind], X_test[:,pas_ind])
    Y_out2 = torch.argmax(nn.functional.softmax(Y_out1, dim=1), dim=1)
    
    Y_out3 = Final_model(X_train[:,act_ind], X_train[:,pas_ind])
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
    return accuracy, accuracy_t