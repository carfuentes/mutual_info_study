from grid_search import gride_function
from scipy.optimize import brute
from parameters import rho, i_scaling, beta_scaling, n_range, nrmse, noise, spline, euler, save, single, tau, c_n, shell


## PARAMETERS
#Data
directory="Dataset1/"
file_path='network_edge_list_ENCODE.csv'
folder="noise"


## RUN
rranges = ((0, 1), (0, 1))
params=directory,file_path,folder, rho, [0], nrmse, noise
minimization=brute(gride_function,rranges,args=params,full_output=True)

## OUTPUT
with open("out_file.txt","w") as f_out:
    f_out.write("### MODELING RESERVOIR WITH DATA FROM %s ###\n\n\n"%file_path)
    f_out.write("## Parameters ##\n")
    f_out.write("# Noise:\n tau=%.2f c_n=%.2f\n"%(tau,c_n))
    f_out.write("# Net: \n i_scaling: %.2f beta_scaling: %.2f rho %.2f\n\n\n"%(i_scaling,beta_scaling,rho))
    f_out.write("## RESULTS ##\n")
    f_out.write("%s\n"%str(minimization))