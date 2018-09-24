from sklearn.model_selection import ParameterGrid
from parameters import rho, n_range, nrmse, noise, spline, euler, save, single, tau, c_n, notebook
from test_functions import test
from graph_analysis import spline_plotting
import numpy as np
import sys

## CLUSTER PROCESSING
#def get_param_fixed_value(cluster_n):
#    return np.array([cluster_n * 0.1])

# Get ranges according to cluster number
#cluster_n=float(sys.argv[1])
#rho_range = beta_range = i_range = np.arange(0.1,1.5,0.1)

#if cluster_n <= 15:
#    rho_range = get_param_fixed_value(cluster_n)
#elif 15 < cluster_n <= 30:
#    i_range= get_param_fixed_value(cluster_n-15)
#elif cluster_n > 30:
#    beta_range= get_param_fixed_value(cluster_n-30)
beta_scaling=np.random.uniform(0,1.5)
rho=np.random.uniform(0,1.5)
i_scaling=np.random.uniform(0,1.5)
print("beta", beta_scaling,"rho",rho,"i_range",i_scaling)
#Data
directory="Dataset1/"
file_path='network_edge_list_EcoCyc.csv'
folder="noise"

## RUN
with open("parameter_range_%.2f-%.2f-%.2f.txt"%(rho,beta_scaling,i_scaling),"w") as f_out:
    f_out.write("### MODELING RESERVOIR WITH DATA FROM %s ###\n\n\n"%file_path)
    f_out.write("## Parameters ##\n")
    f_out.write("# Noise: tau=%.2f c_n=%.2f\n"%(tau,c_n))
    f_out.write("## RESULTS ##\n")
    net,nrmse_by_n,mi_by_n,MImax_n= test(directory,file_path,folder,rho, i_scaling,beta_scaling,n_range, nrmse, noise, euler, save, single, spline)
    nrmse_min=min(nrmse_by_n[0].values())
    FWHM=spline_plotting(mi_by_n[15])
    f_out.write("# rho %.2f, i_scaling: %.2f, beta_scaling: %.2f \n"%(rho,i_scaling,beta_scaling))
    f_out.write("# Minimum nrmse at n=0\n %.20f\n"%(nrmse_min))
    f_out.write("# Maximum FWHM at n=15\n %.20f\n"%(FWHM))