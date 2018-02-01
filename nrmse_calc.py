#imports
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error



#functions
def nrmse(y,x):
    return sqrt(mean_squared_error(x,y))/np.std(x)


def nrmse_n(Yt, Xt,i_max,errorLen,startLen):
    NRMSE_i={}
    for i in range(i_max+1):
        NRMSE_i[i]=nrmse(Yt[0,0:errorLen],Xt[startLen-i:startLen+errorLen-i]) 
    
    return NRMSE_i
