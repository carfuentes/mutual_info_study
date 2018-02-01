from test_functions import test

def gride_function(to_optimize, *params):
    i_scaling, beta_scaling =to_optimize
    directory,file,folder,spectral_radius,n_range, nrmse, noise=params
    net,nrmse_by_n,mi_by_n,MImax_n,FWHM= test(directory,file,folder,spectral_radius, i_scaling,beta_scaling,n_range, nrmse, noise)
    return min(nrmse_by_n[0].values())