# import sys
# a = '\\Federated learning'
# if a not in sys.path:
#     sys.path.append(a)

import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('..')
from code_repository import My_plot
os.chdir(os.getcwd()+'\\LS_Halfstar_LB_UB')

import itertools
marker = itertools.cycle(('o',',', '+', '.',  '*')) 
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
#1: Drive 2: Satellite

%matplotlib qt
fig, axs = plt.subplots(1,3)


STR = [['Bank_', 19, 0], ['Satellite_',36, 1], ['Robot_',24, 2]]
data_name = ['Bank', 'Satellite', 'Robot']
LEG = ['LS UB', 'Half* UB', 'LS', 'LS empirical', 'Half*', 'Hlaf* empirical','LB']
for i in STR:
    
    mark = itertools.cycle(('+',',', 'o', 'd', '*', 'x', 'd')) 
    lst = itertools.cycle(('-','--', '-.', ':'))
    col = itertools.cycle(('b','g', 'r', 'c', 'm', 'y', 'k'))
    
    f = open(i[0]+'UB11', 'rb')
    UB11 = pickle.load(f)
    f.close()
    
    t1, LS_emp = My_plot(i[1], i[0]+'ESA_NoClamp.pckl')
    
    t2, Half_emp = My_plot(i[1], i[0]+'ESA_0centre.pckl')
    
    f = open(i[0]+'UB12', 'rb')
    UB12 = pickle.load(f)
    f.close()
    
    f = open(i[0]+'UB21', 'rb')
    UB21 = pickle.load(f)
    f.close()
    
    f = open(i[0]+'UB22', 'rb')
    UB22 = pickle.load(f)
    f.close()
    
    f = open(i[0]+'LB', 'rb')
    LB = pickle.load(f)
    f.close()
    
    f = open(i[0]+'LS', 'rb')
    LS = pickle.load(f)
    f.close()
    
    f = open(i[0]+'Half_star', 'rb')
    Half_star = pickle.load(f)
    f.close()
    
    N = np.arange(1,len(UB11[0])+1)/i[1]*100
    UB1 = np.minimum(UB11, UB12)
    UB2 = np.minimum(UB21, UB22)
    axs[i[2]].plot(N.flatten(), UB1.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    axs[i[2]].plot(N.flatten(), UB2.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    axs[i[2]].plot(N.flatten(), LS.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    axs[i[2]].plot(t1, LS_emp, linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    axs[i[2]].plot(N.flatten(), Half_star.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    axs[i[2]].plot(t2, Half_emp, linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    axs[i[2]].plot(N.flatten(), LB.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
    
    axs[i[2]].set_ylabel('MSE', fontsize=15)
    axs[i[2]].set_xlabel('Ratio of passive party features ('+i[0][:-1]+')', fontsize=15)
    axs[i[2]].grid()
    if i[2]==2:
        axs[i[2]].legend(LEG, loc='lower right')
    



# plt.figure()

# i = ['Satellite_',36, 0, 1, 0.4]
# # data_name = ['Bank', 'Satellite', 'Robot', 'Synthetic1']
# LEG = ['UB11', 'UB12', 'UB1', 'UB21', 'UB22', 'UB2']

    
# mark = itertools.cycle(('o',',', '+', '.', '*', 'x', 'd')) 
# lst = itertools.cycle(('-','--', '-.', ':'))
# col = itertools.cycle(('b','g', 'r', 'c', 'm', 'y', 'k'))

# f = open(i[0]+'UB11', 'rb')
# UB11 = pickle.load(f)
# f.close()

# f = open(i[0]+'UB12', 'rb')
# UB12 = pickle.load(f)
# f.close()

# f = open(i[0]+'UB21', 'rb')
# UB21 = pickle.load(f)
# f.close()

# f = open(i[0]+'UB22', 'rb')
# UB22 = pickle.load(f)
# f.close()


# N = np.arange(len(UB11[0]))/i[1]
# UB1 = np.minimum(UB11, UB12)
# UB2 = np.minimum(UB21, UB22)
# plt.plot(N.flatten(), UB11.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
# plt.plot(N.flatten(), UB12.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
# plt.plot(N.flatten(), UB1.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
# plt.plot(N.flatten(), UB21.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
# plt.plot(N.flatten(), UB22.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)
# plt.plot(N.flatten(), UB2.flatten(), linestyle=next(lst), marker = next(mark), color = next(col), linewidth=2)


# plt.ylabel('MSE', fontsize=15)
# plt.xlabel('Ratio of passive party features ('+i[0][:-1]+')', fontsize=15)
# plt.grid()
# plt.ylim([0, 0.35])
# plt.legend(LEG, loc='lower right')
# plt.show()
    
    
