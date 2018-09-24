from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV
from parameters import rho, n_range, nrmse, noise, spline, euler, save, single, tau, c_n, notebook
from test_functions import test
from graph_analysis import spline_plotting
import numpy as np
import sys

# Get random numbers
beta=np.random.uniform(0,1.5)
i=np.random.uniform(0,1.5)
rho=np.random.uniform(0,1.5)
print("beta_range", beta,"rho",rho,"i",i)  
#Data
directory="Dataset1/"
file_path='network_edge_list_ENCODE.csv'
folder="noise"

## RUN
with open("rho_%s-beta_%s-i_%s.txt"%(str(rho),str(beta),str(i)),"w") as f_out:
    f_out.write("### MODELING RESERVOIR WITH DATA FROM %s ###\n\n\n"%file_path)
    f_out.write("## Parameters ##\n")
    f_out.write("# Noise:\n tau=%.2f c_n=%.2f\n"%(tau,c_n))
    f_out.write("## RESULTS ##\n")

    net,nrmse_by_n,mi_by_n,MImax_n= test(directory,file_path,folder,rho, i, beta,n_range, nrmse, noise, euler, save, single, spline)
    nrmse_min=min(nrmse_by_n[0].values())
    FWHM=spline_plotting(mi_by_n[15])
    f_out.write("rho %.2f, i_scaling: %.2f, beta_scaling: %.2f \n"%(rho, i, beta))
    f_out.write("Minimum nrmse at n=0 %.20f\n"%(nrmse_min))
    f_out.write("Maximum FWHM at n=15 %.20f\n"%(FWHM))