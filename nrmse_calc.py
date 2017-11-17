#imports
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error



#functions
def nrmse(y,x):
    return sqrt(mean_squared_error(x,y))/np.std(x)


def nrmse_n(Yt, Xt,errorLen,trainLen):
    NRMSE_i={}
    for i in range(51):
        NRMSE_i[i]=nrmse(Yt[0,0:errorLen],Xt[trainLen+1-i:trainLen+errorLen+1-i]) 
    
    return NRMSE_i
