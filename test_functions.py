#imports
from datetime import *
from ESN_class import ESN
from plot_functions import *
from graph_analysis import *
import numpy as np
import os
from parameters import *
from mutual_info import memory_capacity_n
from nrmse_calc import nrmse_n

#functions

    
def printTime(*args):

    print( datetime.now(),"".join(map(str,args)))

def test(directory,file_path, folder, spectral_radius, i_scaling,beta_scaling,n_range, nrmse, noise, euler=True, save=False, single=True, gaussian=False, notebook=True):
    #init
    printTime(file_path)
    filename=file_path[file_path.index("list")+5:file_path.index(".csv")]

    #Run network
    printTime("Running network...")
    net=ESN(os.path.join(directory, file_path),1,1,spectral_radius)
    np.random.seed(42)
    net.initialize(i_scaling,beta_scaling)
    printTime("SR", net.spectral_radius)
    
    #Choose input and collect states
    net.collect_states_ODEINT_NOISE(initLen, trainLen, testLen, euler, noise,dt=0.001)
    #X=net.u
    printTime(net.u.shape)
    
    #plot reservoir units activations
    figure(num=None, figsize=(20, 4), dpi=80, facecolor='w', edgecolor='k')
    title("Reservoir activations")
    plot(net.X[2:20,500:1000].T)
    
    #train for n steps delay
    printTime("Training network...")
    Y_n={}
    mi_by_n={}
    nrmse_by_n={}
    MImax_n={}
    figure(figsize=(30, 5),dpi=80, facecolor='w', edgecolor='k')
    
    
    for index,n in enumerate(n_range):
        printTime("n=",n)
        
        net.calculate_weights_derivative_OLD(initLen,trainLen,n)
        net.run_predictive_derivative_OLD(testLen,trainLen)
    
        if noise=='ahora no' and n==0:
            printTime("Autocorrelation of predicted noise")
            autocorr=autocorrelation(net.Y.reshape(net.Y.shape[1]))
            exponential_fitting(autocorr,exp_func)
            
        #mi_by_n[n]=memory_capacity_n(net.Y, net.u,(subsetLen-initLen)*m,500,i_max)
        mi_by_n[n]=memory_capacity_n(net.Y, net.u,trainLen,miLen,i_max,n_max)
        #nrmse_by_n[n]=nrmse_n(net.Y,net.u,i_max,errorLen,(subsetLen-initLen)*m)
        nrmse_by_n[n]=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen,n_max)
        
        #Plots
        printTime("%d trained to n =%d delay FINISHED" %(net.res_size,n))
        
        
        Y=net.Y
        Y_n[n]=Y
    
    get_figure_noise(net,Y_n)
    plot_nrmse_mi_by_n(n_range,nrmse_by_n,MImax_n,mi_by_n)
    show()
    
    

       
    return net,Y_n,nrmse_by_n,mi_by_n,MImax_n