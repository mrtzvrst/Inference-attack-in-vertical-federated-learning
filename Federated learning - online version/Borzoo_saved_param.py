from scipy.io import savemat
import torch
import numpy as np
import pickle

f = open('LR_model_bank.pt', 'rb')
model_data = pickle.load(f)
f.close()

model = model_data[0]
X = model_data[3]
Weights = np.array(list(model.named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model.named_parameters())[1][1].detach().cpu(), dtype = np.float64)

savemat("Weights.mat", {'Weights': np.array(Weights)})
savemat("Biases.mat", {'Biases': np.array(Biases)})





filename = 'bank-additional-full.csv'
with open(filename) as f:
    lines = f.readlines()
X, Y = [], []
for line in lines:
    temp = list(line.replace("\"","").strip().split(';')) 
    X += [temp]
tt = X[0]
X = X[1:]
for i in range(len(X)):
    X[i][20] = 0 if X[i][20]=='no' else 1
    
for i in range(20):
    if X[0][i].replace('.','',1).isdigit():
        MIN, MAX = float('inf'), -float('inf')
        for j in range(len(X)):
            X[j][i] = float(X[j][i])
            MIN = min(MIN, X[j][i])
            MAX = max(MAX, X[j][i])
        print(tt[i],f': numerical feature with max:{MAX} and min:{MIN}')
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
            
        print(tt[i],': categorical feature:', np.round(np.array([n for n, m in Dict0.values()]),decimals=3))
        
        for j in range(len(X)):
            X[j][i] = Dict0[X[j][i]][0]
X = np.array(X)
