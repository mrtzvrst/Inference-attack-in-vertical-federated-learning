import numpy as np
from torch import nn, optim
from My_centralized_codes import random_mini_batches, Read_data, Log_Reg, NeuralNet_Tanh, NeuralNet_Relu, Outlier_detection
import pickle
import matplotlib.pyplot as plt


fig, axs = plt.subplots(1,4)


f = open('LR_model_bank.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[0].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[0].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-+')
axs[0].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[0].set_title('bank')


f = open('LR_model_robot.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[1].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[1].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-+')
axs[1].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[1].set_title('robot')

f = open('LR_model_Satellite.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[2].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[2].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-+')
axs[2].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[2].set_title('satellite')

f = open('LR_model_Syn_v1.pt', 'rb')
model_data = pickle.load(f)
f.close()
X = model_data[3]
X_m = np.mean(X, axis=0)
X_v = np.var(X, axis=0)
axs[3].plot(np.arange(1,X_m.shape[0]+1), X_m,'-+')
axs[3].plot(np.arange(1,X_m.shape[0]+1), np.ones((X_m.shape[0]))*np.mean(X_m),'-+')
axs[3].plot(np.arange(1,X_v.shape[0]+1), X_v,'-o')
axs[3].set_title('Synthetic')