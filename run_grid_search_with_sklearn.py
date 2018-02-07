from sklearn.model_selection import ParameterGrid
from parameters import rho, n_range, nrmse, noise, spline, euler, save, single, tau, c_n, notebook
from test_functions import test
from graph_analysis import spline_plotting
import numpy as np
# PARAMETERS

#Data
directory="Dataset1/"
file_path='network_edge_list_ENCODE.csv'
folder="noise"

#CREATE DE GRID
param_grid = {'i_scaling': np.arange(0.01,1.5,0.1), 'beta_scaling' : np.arange(0.01,1.5,0.1), "rho": np.arange(0.01,1.5,0.1)}
grid = ParameterGrid(param_grid)
nrmse_min_dict={}
FWHM_dict={}
## RUN
with open("out_file.txt","w") as f_out:
    f_out.write("### MODELING RESERVOIR WITH DATA FROM %s ###\n\n\n"%file_path)
    f_out.write("## Parameters ##\n")
    f_out.write("# Noise:\n tau=%.2f c_n=%.2f\n"%(tau,c_n))
    f_out.write("## RESULTS ##\n")

    for param in grid:
        net,nrmse_by_n,mi_by_n,MImax_n= test(directory,file_path,folder,param["rho"], param["i_scaling"],param["beta_scaling"],n_range, nrmse, noise, euler, save, single, spline)
        nrmse_min=min(nrmse_by_n[0].values())
        FWHM=spline_plotting(mi_by_n[15])
        f_out.write("rho %.2f, i_scaling: %.2f, beta_scaling: %.2f \n"%(param["rho"],param["i_scaling"],param["beta_scaling"]))
        f_out.write("Minimum nrmse at n=0 %.20f\n"%(nrmse_min))
        nrmse_min_dict[str(param["rho"])+"-"+str(param["i_scaling"])+"-"+str(param["beta_scaling"])]=nrmse_min
        FWHM_dict[str(param["rho"])+"-"+str(param["i_scaling"])+"-"+str(param["beta_scaling"])]=FWHM
    f_out.write("Parameters with the minimum NRMSE: %s => Value= %.20f\n"%(min(nrmse_min_dict),min(nrmse_min_dict.values()) ))
    f_out.write("Parameters with the maximum FWHM: %s => Value= %.20f\n"%(max(FWHM_dict),max(FWHM_dict.values())))