from test_functions import test
from parameters import *

## PARAMETERS

#Data
directory="Dataset1/"
file_path='network_edge_list_ENCODE.csv'
folder="noise"


## RUN
net,Y_n,nrmse_by_n,mi_by_n,MImax_n= test(directory,file_path, folder, rho, i_scaling,beta_scaling,n_range, nrmse, noise, euler=True, save=False, single=True, gaussian=False, notebook=True)

## OUTPUT
with open("out_file.txt","w") as f_out:
    f_out.write("### MODELING RESERVOIR WITH DATA FROM %s ###\n\n\n"%file_path)
    f_out.write("## Parameters ##\n")
    f_out.write("# Noise:\n tau=%.2f c_n=%.2f\n"%(tau,c_n))
    f_out.write("# Net: \n i_scaling: %.2f beta_scaling: %.2f rho %.2f\n\n\n"%(i_scaling,beta_scaling,rho))
    f_out.write("## RESULTS ##\n")
    f_out.write("Nrmse: %s\n"%(nrmse_by_n))
    f_out.write("MI: %s \n"% (mi_by_n))