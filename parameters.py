#noise parameters
tau=10
c_n=0.01

#Time lengths
errorLen = 500
trainLen=10000
testLen=1000
initLen=200
subsetLen=1200
m=10
startLen=(subsetLen-initLen)*m

#Net parameters
rho=1
i_scaling=0.5
beta_scaling=0.8
n_range=list(range(100))
i_max=200

#Test parametes
nrmse= True
noise=True
spline=True
euler=True
save=False
single=True
notebook=False




csv_files=['network_edge_list_ENCODE.csv', 'network_edge_list_modENCODE.csv', 'network_edge_list_YEASTRACT.csv', 'network_edge_list_EcoCyc.csv', 'network_edge_list_DBTBS.csv']