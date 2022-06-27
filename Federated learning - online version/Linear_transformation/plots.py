
import pickle
import numpy as np
import matplotlib.pyplot as plt
from LT_codes import My_plot
import itertools
marker = itertools.cycle(('o',',', '+', '.',  '*')) 
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
#1: Drive 2: Satellite

d_tot = 5
STR = 'Bef_Trans_'


LOSS = [#'RG', 
        #'Zero',
        #'Half', 
        
        # 'ESA_NoClamp',
        # 'ESA_Clamp',
        # 'ESA_Clamp_Extended', 
        # 'ESA_0centre', 
        # 'ESA_RCC', 
        'ESA_CLS'
        ]

for i in LOSS:
    filename = STR+i+'.pckl'
    t1, MSE_plt = My_plot(d_tot, filename)
    plt.plot(t1, MSE_plt, ".-", marker = next(marker), label=i)


LOSS1 = [#'RG', 
        #'Zero',
        #'Half', 
        
        # 'ESA_NoClamp', 'ESA_NoClamp_Aft',
        # 'ESA_Clamp', 'ESA_Clamp_Aft',
        # 'ESA_Clamp_Extended', 'ESA_Clamp_Extended_Aft',
        # 'ESA_0centre', 'ESA_0centre_Aft',
        # 'ESA_RCC', 'ESA_RCC_Aft',
        'ESA_CLS', 'ESA_CLS_Aft',
        ]
d_tot = 5
STR = 'Aft_Trans_'
for i in LOSS:
    filename = STR+i+'.pckl'
    t1, MSE_plt = My_plot(d_tot, filename)
    plt.plot(t1, MSE_plt, ".-", marker = next(marker), label=i)
    
plt.legend(LOSS1, bbox_to_anchor=(1, 0.5))


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
