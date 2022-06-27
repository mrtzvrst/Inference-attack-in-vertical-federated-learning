import sys
a = 'C:\\Users\\mrtzv\\Desktop\\leetcode\\Data_disclosure_Borzoo\\Federated learning'
if a not in sys.path:
    sys.path.append(a)
    
import pickle
import numpy as np
import matplotlib.pyplot as plt
from My_centralized_codes import My_plot
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
#1: Drive 2: Satellite

'''drive: 24, satellite: 36, Bank: 19, Robot: 24'''
d_tot = 19
'''Satellite_, Bank_, Robot_'''
STR = 'Bank_'


LOSS = ['ESA_Clamp', 'ESA_Clamp_Extended', #'ESA_NoClamp',
        #'zero', 'half', 'random'
        ]

for i in LOSS:
    filename = 'Accuracy_'+STR+i+'.pckl'
    t1, MSE_plt = My_plot(d_tot, filename)
    plt.plot(t1, MSE_plt, marker = next(marker), label=i)
plt.legend(LOSS, bbox_to_anchor=(1, 0.5))
plt.show()

    
 

    
    
    
    
# f = open('Satellite_Bor1_NoClamp_'+str(3)+'.pckl', 'rb')#########
# MSE = np.array(pickle.load(f))
# f.close()
# for i in range(2,4):##########
#     f = open('Satellite_KLD1_NoClamp_'+str(i)+'.pckl', 'rb')
#     MSE[:,2] += np.array(pickle.load(f))[:,2]
#     f.close()
# MSE[:,2]/=3######

            
# f = open('Satellite_KLD1_NoClamp.pckl', 'wb')
# pickle.dump(MSE, f)
# f.close()
