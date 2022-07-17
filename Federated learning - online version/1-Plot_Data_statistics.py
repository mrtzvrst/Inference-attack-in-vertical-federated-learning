import numpy as np
from torch import nn, optim
from My_centralized_codes import random_mini_batches, Read_data, Log_Reg, NeuralNet_Tanh, NeuralNet_Relu, Outlier_detection
import pickle
import matplotlib.pyplot as plt

%matplotlib qt
fig, axs = plt.subplots(2,2)


f = open('LR_model_bank.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[0][0].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[0][0].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-.')
axs[0][0].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[0][0].set_title('bank')
axs[0][0].set_xlabel('Feature indices', fontsize=15)
axs[0][0].grid()

f = open('LR_model_robot.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[0][1].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[0][1].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-.')
axs[0][1].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[0][1].set_title('robot')
axs[0][1].set_xlabel('Feature indices', fontsize=15)
axs[0][1].grid()


f = open('LR_model_Satellite.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[1][0].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[1][0].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-.')
axs[1][0].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[1][0].set_title('satellite')
axs[1][0].set_xlabel('Feature indices', fontsize=15)
axs[1][0].grid()


f = open('LR_model_Syn_v1.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[1][1].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[1][1].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-.')
axs[1][1].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[1][1].set_title('Synthetic')
axs[1][1].set_xlabel('Feature indices', fontsize=15)
axs[1][1].grid()


plt.legend(['Means of features','Mean of means','Variance of features'])
plt.savefig('Plot_Data_statistics.eps', format='eps')