#imports
from ESN_class import ESN
from plot_functions import *
from graph_analysis import *
import numpy as np
import os
from parameters import tau,i_max,errorLen, trainLen, testLen, initLen, subsetLen, m, startLen
from mutual_info import memory_capacity_n
from nrmse_calc import nrmse_n

#functions
def test(directory,file_path, folder, spectral_radius, i_scaling,beta_scaling,n_range, nrmse, noise, euler=True, save=False, single=True, gaussian=False, notebook=False):
    #init
    print(file_path)
    filename=file_path[file_path.index("list")+5:file_path.index(".csv")]

    #Run network
    print("Running network...")
    net=ESN(os.path.join(directory, file_path),1,1,spectral_radius)
    np.random.seed(42)
    net.initialize(i_scaling,beta_scaling)
    print("SR", net.spectral_radius)
    
    #Choose input and collect states
    net.collect_states_in_subsets(m, initLen, subsetLen, testLen, euler, noise,dt=0.001)
    #X=net.u
    #print(X.shape)

    if notebook:
        #plot reservoir units activations
        figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        title("Reservoir activations")
        plot(net.X[2:20,100:200].T)
        
        if noise:
            print("Autocorrelation of generated noise")
            autocorr=autocorrelation(net.u)
            exponential_fitting(autocorr)
    
    #train for n steps delay
    print("Training network...")
    Y_n={}
    mi_by_n={}
    nrmse_by_n={}
    MImax_n={}
    for n in n_range:
        print("n=",n)
        net.calculate_weights_derivative(initLen,subsetLen,m,n)
        net.run_predictive_derivative(testLen,subsetLen,initLen,m)
    
        if noise and n==0 and notebook:
            print("Autocorrelation of predicted noise")
            autocorr=autocorrelation(net.Y.reshape(net.Y.shape[1]))
            exponential_fitting(autocorr)
            
        mi_by_n[n]=memory_capacity_n(net.Y, net.u,(subsetLen-initLen)*m,500)
        nrmse_by_n[n]=nrmse_n(net.Y,net.u,i_max,errorLen,(subsetLen-initLen)*m)
        
        #Plots
        print("%d trained to n =%d delay FINISHED" %(net.res_size,n))
        
        if nrmse:
            plot_dict_i(filename, nrmse_by_n[n],MImax_n,n,nrmse,single,notebook)
        else:
            plot_dict_i(filename, mi_by_n[n],MImax_n,n,nrmse,single,notebook)
        
        if notebook:
            figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plot_temporal_lines(net.u,net.Y, n,testLen-initLen,filename, tau, folder, save)
            
            figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plot_temporal_lines(net.u,net.Y, n, 50,filename, tau, folder ,save)
        
        if not single:
            Y=net.Y
            Y_n[n]=Y

    if single:
        return net,nrmse_by_n,mi_by_n,MImax_n
    if not single:
        X=net.u
        return net,X,Y_n,nrmse_by_n,mi_by_n,filename,MImax_n


def test_all(directory,folder,spectral_radius, n_range, nrmse, noise, euler=True, save=False):
    #init
    csv_files= [file_path for file_path in os.listdir(directory) if file_path.startswith("network_edge_list")]
    Y_by_file={}
    X_by_file={}
    MI_by_file={}
    NRMSE_by_file={}
    
    for file_path in csv_files:
        net,X,Y,nrmse_by_n,mi_by_n,filename= test(directory,file_path,folder,spectral_radius, n_range, nrmse, noise, euler, save, single=False)
        X_by_file[filename]=X
        Y_by_file[filename]=Y
        MI_by_file[filename]=mi_by_n
        NRMSE_by_file[filename]=nrmse_by_n
        
        #plots
    for n in n_range:    
        figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        title("n="+str(n))
        if nrmse:
            plot_dict_by_file(NRMSE_by_file,n,tau,folder, nrmse, save)
        else:
            plot_dict_by_file(MI_by_file,n,tau,folder, nrmse, save)
        
    return X_by_file, Y_by_file, NRMSE_by_file, MI_by_file