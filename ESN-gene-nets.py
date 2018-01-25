
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
from matplotlib.pyplot import *
from sklearn.metrics import mean_squared_error
import os
import scipy
from math import sqrt
import numba
from numba import jitclass
from numba import int32, float32, int64, float64
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import UnivariateSpline
#from minepy import MINE


# In[2]:


from preprocessing_net import get_cyclic_net
from mutual_info import memory_capacity_n
from nrmse_calc import nrmse, nrmse_n


# In[3]:


#imports
import entropy_estimators as ee
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import random as rand
import scipy


# In[4]:


#ESN_class.py imports
import networkx as nx
import numpy as np
from sampling import random_sampling_normal_from_range
import random as rand
import scipy
from preprocessing_net import get_cyclic_net


# In[5]:


@numba.jit(nopython=True)
def colored_noise_euler_integration(res_size,decay,Win,W,x_0, u_0, tau, c, t_stop, dt=0.001):
    np.random.seed(42)
    mu=np.exp(-dt/tau)
    sigma= sqrt( ((c * tau)/2) * (1-mu**2) )
    
    
    t = np.linspace(0, t_stop, int(t_stop/dt))
    
    x=np.zeros((len(t),res_size))
    x[0,:]=x_0

    u = u_0 * np.ones_like(t)
    
    for i in range(0, len(t) - 1):
        u[i+1] = u[i]* mu + sigma * np.random.normal()
        x[i+1,:] = x[i,:] + dt * dx_act_dt(x[i,:], u[i],res_size,decay,Win,W)
            
    return u,x


# In[6]:


@numba.jit(nopython=True)
def dx_act_dt(x,u,res_size,decay,Win,W):
        x=x.reshape(res_size,1)
        x_act=decay*0.5*(np.tanh( np.dot( Win, np.vstack((np.array(1),np.array(u)) )) + np.dot( W, x ) )+1) - (decay * x)
        return x_act.reshape(res_size)


# In[56]:


#@jitclass(spec,nopython=True)
class ESN(object):
    def __init__(self, filename, in_size, out_size, spectral_radius):
        self.res_size= self.build_adj_weighted_matrix(filename).shape[0]
        self.in_size=in_size
        self.out_size=out_size
        self.spectral_radius= spectral_radius
        self.W0=self.build_adj_weighted_matrix(filename)
        self.W=None
        self.Win=None
        self.Wout=None
        self.X=None
        self.Y=None
        self.x=np.zeros((self.res_size,1))
        self.x0_e=np.random.rand(self.res_size)
        self.x0=np.insert(np.random.rand(self.res_size)*10,0,[1.0,1.0,1.0])
        self.u0=0
        self.decay=np.random.rand(self.res_size).reshape((self.res_size,1))
        self.u=None
        self.x_act=None

   
    def build_adj_weighted_matrix(self, filename):
        #NETWORK v2.0
        net=get_cyclic_net(filename)
        for edge in net.edges(data="mode", default=0):
            source,target,mode=edge
            if mode== "+":
                net[source][target]["weight"]= rand.gauss(0,1)*1
            elif mode== "-":
                net[source][target]["weight"]= rand.gauss(0,1)*-1
            elif mode== 0:
                net[source][target]["weight"]= rand.gauss(0,1)
        return nx.to_numpy_matrix(net)
    
    def initialize(self): 
        np.random.seed(42)
        self.Win=np.random.normal(size=(self.res_size,1+self.in_size))*0.3
        self.Win[:,0]=np.random.normal(size=(self.res_size,))*0.2
        self.W0 = np.squeeze(np.asarray(self.W0))
        radius = np.max(np.abs(np.linalg.eigvals(self.W0)))
        self.W= (self.spectral_radius/radius)*self.W0
        return self.X
    
    def dx_dt(self, uyz_x,t,a,b,c,decay):
        u=uyz_x[0]
        y=uyz_x[1]
        z=uyz_x[2]
        x=uyz_x[3:]
        
        du_dt=-z-y
        dy_dt=u+a*y
        dz_dt=b+z*(u-c)
        dx_dt=self.dx_act_dt(x,u**2)
        return np.insert(dx_dt,0,[du_dt,dy_dt,dz_dt])
    
    def du_dt_rossler(self,z,y):
        return -z-y
    
    def dy_dt_rossler(self, u, a, y):
        return u+a*y
    
    def dz_dt_rossler(self, b, z, u, c):
        return b+z*(u-c)
    
    def dx_act_dt(self, x,u):
        return dx_act_dt(self.res_size,self.decay,self.Win,self.W, x,u)
    
    def colored_noise_euler_integration(self, x_0, u_0, tau, c, t_stop, dt=0.001):
        return colored_noise_euler_integration(self.res_size,self.decay,self.Win,self.W, x_0, u_0, tau, c, t_stop, dt=0.001)
    
    def rossler_euler_integration(self, x_0, u_0, a, b, c, t_stop, dt=0.001):
        t = np.linspace(0, t_stop, int(t_stop/dt))
    
        x=np.zeros((len(t),self.res_size))
        x[0,:]=x_0

        u = u_0 * np.ones_like(t)
        y = u_0 * np.ones_like(t)
        z = u_0 * np.ones_like(t)
    
        for i in range(0, len(t) - 1):
            u[i+1] = u[i] + dt * self.du_dt_rossler(z[i],y[i])
            y[i+1] = y[i] + dt * self.dy_dt_rossler(u[i],a,y[i])
            z[i+1]= z[i] + dt * self.dz_dt_rossler(b,z[i],u[i],c)
            x[i+1,:] = x[i,:] + dt * self.dx_act_dt(x[i,:], u[i]**2)
            
        return u,x
    
    
    def collect_states(self, data, init_len, train_len, a=0.3):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        for t in range(train_len):
            u = data[t]
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            if t >= init_len:
                self.X[:,t-init_len]= np.vstack((1,u,self.x))[:,0]
        
        return self.X
    
    def collect_states_in_subsets(self, m, init_len, subset_len, test_len, euler, noise,dt=0.001):
        self.X=np.zeros((self.res_size+self.in_size+1, 1))
        self.u=np.array([])
        self.x_act=np.zeros((1,self.res_size))
        print(self.x_act.shape)
        print(self.X.shape)
        for subset in range(m):
            print("Subset ", str(subset+1))
            X, u, x_act= self.collect_states_derivative(init_len,subset_len,euler,noise,dt)
            print(x_act.shape)
            print(X.shape)
            self.X=np.concatenate((self.X,X),axis=1)
            self.u=np.append(self.u,u)
            self.x_act=np.concatenate((self.x_act,x_act),axis=0)
            
        print("Test subset")
        X, u, x_act= self.collect_states_derivative(init_len,test_len,euler,noise,dt)
        #self.X=np.concatenate((self.X,X),axis=1)
        self.u=np.append(self.u,u)
        self.x_act=np.concatenate((self.x_act,x_act),axis=0)
        self.X=self.X[:,1:]
        self.x_act=self.x_act[1:,:]
    
    def collect_states_derivative(self, init_len, train_len, euler, noise, dt=0.001):
        X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t_stop=train_len
            
        if not euler:
            print("Collecting states with rossler input using odeint built in...")
            print("Parameters: a= %.2f b= %.2f c= %.2f"%(a,b,c))
            t=np.arange(train_len+test_len)
            uyz_x=scipy.integrate.odeint(self.dx_dt,self.x0,t,args=(a,b,c,self.decay))
            u=uyz_x[:,0]**2
            x_act=uyz_x[:,3:]
            
        else:
            
            if noise:
                print("Collecting states with noise input...")
                print("Parameters: tau= %.2f c_n= %.2f"%(tau,c_n))
                u, x_act=self.colored_noise_euler_integration(self.x0_e, self.u0, tau, c_n, t_stop, dt)
                
            else:
                print("Collecting states with rossler input using euler integration...")
                print("Parameters: a= %.2f b= %.2f c= %.2f"%(a,b,c))
                u, x_act=self.rossler_euler_integration(self.x0_e, self.u0, a,b, c, t_stop, dt)

            indexes=[int(t/dt) for t in range(0,t_stop)]
            u=u[indexes]
            x_act=x_act[indexes,:]
   
        for t in range(init_len,train_len):
            x_concat=x_act[t,:].reshape(x_act[t,:].shape[0],1)
            u_concat=u[t]
            X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return X, u[init_len:], x_act[init_len:,:]
     
        
    def calculate_weights(self, data, init_len, train_len,beta=1e-8 ):
        Y=data[None,init_len+1:train_len+1]
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1)))
        return self.Wout
    
    def calculate_weights_derivative(self,init_len, subset_len, m, n, beta=1e-8 ):
        start_len= (subset_len-init_len) * m
        Y=np.array([self.u[init_len-n:start_len-n]])
        X=self.X[:,init_len:]
        X_T=X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(X,X_T) + beta * np.eye(self.res_size+self.in_size+1))) #w= y*x_t*(x*x_t + beta*I)^-1
        return self.Wout
    
    def run_predictive_derivative(self, test_len, subset_len,init_len,m):
        self.Y = np.zeros((self.out_size,test_len))
        start_len= (subset_len-init_len) * m
        for t in range(start_len,start_len+(test_len-init_len)):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            y = np.dot( self.Wout, np.vstack((1,u_concat,x_concat)) )
            self.Y[:,t-start_len] = y
           
        
        return self.Y


# In[8]:


##################################################################################


# In[9]:


#                                   FUNCTIONS                                    #


# In[33]:


def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    result=result[int(result.size/2):]
    return result/result[0]


# In[34]:


def testing_gene_net(directory,input_data,data):
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    print(csv_files)
    MI_by_file={}
    for file in csv_files:
        filename=file[file.index("list")+5:file.index(".csv")]
        net=ESN(os.path.join(directory, file),1,1,0.95)
        net.initialize()
        net.collect_states(input_data,initLen,trainLen)
        net.calculate_weights(input_data,initLen,trainLen)
        net.run_generative(input_data,testLen,trainLen)
      
        MI_by_file[filename]=memory_capacity_n(net.Y, data,100)
        nrmse= sqrt(mean_squared_error(data[trainLen+1:trainLen+errorLen+1],net.Y[0,0:errorLen])/np.std(net.Y[0,0:errorLen]))
        print(net.res_size, 'NRMSE = ' + str( nrmse ))
        print(memory_capacity_n(net.Y, data,20))
        
    return MI_by_file


# In[35]:


def testing_gene_net_file(directory,file):
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    net=ESN(os.path.join(directory, file),1,1,0.95)
    net.initialize()
    net.collect_states(data,initLen,trainLen)
    net.calculate_weights(data,initLen,trainLen)
    net.run_generative(data,testLen,trainLen)
    nrmse= sqrt(mean_squared_error(data[trainLen+1:trainLen+errorLen+1],net.Y[0,0:errorLen])/np.std(net.Y[0,0:errorLen]))
    print(net.res_size, 'NRMSE = ' + str( nrmse ))
    return memory_capacity_n(net.Y, data,100)


# In[36]:


def test_all(directory,folder,spectral_radius, n_range, nrmse, noise, euler=True, save=False):
    #init
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    Y_by_file={}
    X_by_file={}
    MI_by_file={}
    NRMSE_by_file={}
    
    for file in csv_files:
        net,X,Y,nrmse_by_n,mi_by_n,filename= test(directory,file,folder,spectral_radius, n_range, nrmse, noise, euler, save, single=False)
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
  


# In[37]:


def test(directory,file,folder,spectral_radius, n_range, nrmse, noise, euler=True, save=False, single=True, gaussian=False):
    #init
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    
    #Run network
    print("Running network...")
    net=ESN(os.path.join(directory, file),1,1,spectral_radius)
    np.random.seed(42)
    net.initialize()
    print("SR", net.spectral_radius)
    
    #Choose input and collect states
    net.collect_states_in_subsets(m, initLen, subsetLen, testLen, euler, noise,dt=0.001)
    X=net.u
    print(X.shape)
    
    if noise:
        print("Autocorrelation of generated noise")
        autocorr=autocorrelation(X)
        exponential_fitting(autocorr)
    
    #plot reservoir units activations
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("Reservoir activations")
    plot(net.X[2:20,100:200].T)
    show()
    
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
    
        #Calculate output
        Y=net.Y
        Y_n[n]=Y
        
        if noise and n==0:
            print("Autocorrelation of predicted noise")
            autocorr=autocorrelation(Y.reshape(Y.shape[1]))
            exponential_fitting(autocorr)
            
        mi_by_n[n]=memory_capacity_n(net.Y, net.u,(subsetLen-initLen)*m,500)
        nrmse_by_n[n]=nrmse_n(net.Y,net.u,i_max,errorLen,(subsetLen-initLen)*m)
        
        #Plots
        print("%d trained to n =%d delay FINISHED" %(net.res_size,n))
        
        if single:
            figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            title("n="+str(n))

            if nrmse:
                print("Plot nrmse")
                plot_dict_i(filename, nrmse_by_n[n],MImax_n,n)
                if save:
                    savefig("plots/folder/%s/%s_n%d_tau_%d" %(folder,filename,n,tau))
            else:
                print("Plot mi")
                FWHM=plot_dict_i(filename, mi_by_n[n],MImax_n,n,nrmse=False,gaussian=gaussian)
                
            show()

            figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plot_temporal_lines(X,Y, n,testLen-initLen,filename, tau, folder, save)

            figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plot_temporal_lines(X,Y, n, 50,filename, tau, folder ,save)

    
    return net,X,Y_n,nrmse_by_n,mi_by_n,filename,MImax_n,FWHM


# In[38]:


def plot_dict_i(key,dict_i,MImax_n,n,nrmse=True, single=True, gaussian=False):
    x=[]
    y=[]
    FWHM=0
    
    if nrmse:
        point=min(dict_i, key=dict_i.get)
        print(key,"=> min=",point, "value", dict_i[point])
    else:
        point=max(dict_i, key=lambda y: abs(dict_i[y]))
        print(key,"=> max=",point)
        MImax_n[n]=abs(dict_i[point])
        
    for i,value in dict_i.items():
        x.append(i)
        y.append(value)
    
    plot(x,y,"-o",label=key)
    plot(point,dict_i[point],marker='o')
    
    
    if single:
        xlabel("i=[0,%d]"%(x[-1]))
        
        if nrmse:
            ylabel("NRMSE(X(t-i), Y(t))")
        else:
            ylabel("MI(X(t-i), Y(t))")
    if gaussian:
        FWHM = spline_plotting(x,y)
    
    return FWHM


# In[39]:


def plot_dict_by_file(dict_by_file,n,tau,folder,nrmse=True,save=True):
    for file in dict_by_file.keys():
        plot_dict_i(file, dict_by_file[file][n],nrmse=nrmse,single=False)
    legend(loc='upper left')
    xlabel("i=[0,%d]"%(len(dict_by_file[file][n].keys())-1))
    
    if nrmse:
        ylabel("NRMSE(X(t-i), Y(t))")
    else:
        ylabel("MI(X(t-i), Y(t))")
        
    if save:
        savefig("plots/nrmse_i_all_files/%s/nrmse_all_n%d_tau_%s" %(folder,n,str(tau)))
    show()
    


# In[40]:


def plot_temporal_lines(u,Y,n,length,filename, tau,folder,save=True):
    plot( u[startLen-n:startLen+length-n], 'g' )
    plot( Y.T[0:length], 'b' )
    xlabel("time")
    ylabel("signal")
    title('Target and generated signals $y(t)$ starting at $t=0$ until $t=%s$ with a delay of n=%d ' %(length,n))
    legend(['Target signal', 'Free-running predicted signal'])
    if save:
        savefig("plots/input-vs-output/%s/%s_len%d_n%d_tau%s" %(folder,filename,length,n,str(tau)))
    show()


# In[41]:


def func(x, a, b, c):
    return a * np.exp(-1/b * x) + c


# In[42]:


def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


# In[43]:


def exponential_fitting(x,MI=False):
    if MI: #x is MI
        xdata=np.array(list(x.keys()))
        ydata=np.array(list(x.values()))
    else: #x is autocorr
        a=x[:np.argmax(x<0)]
        ydata=a
        xdata=np.arange(a.shape[0])
    popt, pcov = curve_fit(func, xdata, ydata)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot(xdata, ydata, '-o', label='data')
    plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    xlabel('n')
    ylabel('MImax')
    legend()
    show()
    return popt


# In[44]:


def gaussian_fitting(x,y):
    xdata=ar(x)
    ydata=ar(y)
    n = len(xdata)                          
    mean = sum(xdata*ydata)/n                  
    sigma = sum(ydata*(xdata-mean)**2)/n  
    popt, pcov = curve_fit(gaus, xdata, ydata,p0=[1,mean,sigma])
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot(xdata, ydata, '-o', label='data')
    plot(xdata, gaus(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    xlabel('n')
    ylabel('MImax')
    legend()
    show()


# In[45]:


def spline_plotting(x,y):
    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM=abs(r1-r2)
    print("r1=%f and r2=%f"%(r1,r2))
    print("FWHM",FWHM)
    plot(x, y)
    plot(r1,0,marker='o')
    plot(r2,0,marker='o')
    axvspan(r1, r2, facecolor='g', alpha=0.5)
    show()
    return FWHM


# In[46]:


##################################################################################


# In[47]:


#                                  PARAMETERS                                    #


# In[48]:


# TRAINING AND TEST LENGHT
errorLen = 500
trainLen=10000
testLen=1000
initLen=200
subsetLen=1200
m=10
startLen=(subsetLen-initLen)*m
#Files
csv_files=['network_edge_list_ENCODE.csv', 'network_edge_list_modENCODE.csv', 'network_edge_list_YEASTRACT.csv', 'network_edge_list_EcoCyc.csv', 'network_edge_list_DBTBS.csv']


# In[49]:


##################################################################################


# In[50]:


#                                   TESTEOS                                      #


# In[51]:


# TESTEO get_cyclic_net
G=get_cyclic_net(os.path.join("Dataset1/", "network_edge_list_modENCODE.csv"))
len(G.nodes())


# In[29]:


#TESTEO adjacency matrix
net=ESN(os.path.join("Dataset1/", "network_edge_list_DBTBS.csv"),1,1,0.95)
net.W0


# In[30]:


#TESTEO initialize
net.initialize()
print(net.W.shape)
print(max(abs(scipy.linalg.eig(net.W)[0])))


# In[31]:


#TESTEO collect states
net.collect_states_derivative(a,b,c, initLen, trainLen, testLen)
net.X.shape
net.X[:,7]


# In[ ]:


##################################################################################


# In[ ]:


## NUMBA DECORATIONS


# In[ ]:


net=ESN(os.path.join("Dataset1/", "network_edge_list_DBTBS.csv"),1,1,0.95)
net.initialize()
for attr,value in net.__dict__.items():
    print(attr,numba.typeof(value))


# In[ ]:


##################################################################################


# In[ ]:


#                             RESULTS  NOISE                                         #


# In[52]:


# SINGLE FILE:
file=csv_files[0]

## N range
n_range=list(range(100))
i_max=80


# In[27]:


# COLORED NOISE


# In[53]:


## Parameters
tau=1
c_n=1


# In[35]:


#Nrmse
net,X,Y,nrmse_i,mi_i,filename,MImax_n=test("Dataset1",file, "noise", 0.95, n_range, nrmse=True, noise=True)


# In[128]:


#MI
net,X,Y,nrmse_i,mi_i,filename, MImax_n_tau_1,FWHM=test("Dataset1",file, "noise", 0.95, n_range, nrmse= False, noise=True)
exponential_fitting(MImax_n_tau_1,MI=True)


# In[62]:


## Parameters
tau=10
c_n=1


# In[57]:


#Mi
net,X,Y,nrmse_i,mi_i,filename, MImax_n_tau_100,FWHM=test("Dataset1",file, "noise", 0.95, n_range, nrmse= False, noise=True)
exponential_fitting(MImax_n_tau_100, MI=True)


# In[ ]:


#MI for a tau range


# In[ ]:


tau_range=list(range(1,11))
MImax_n_1=[]
for tau in range(1,11):
    net,X,Y,nrmse_i,mi_i,filename, MImax_n,FWHM=test("Dataset1",file, "noise", 0.95, n_range, nrmse= False, noise=True)
    exponential_fitting(MImax_n, MI=True)
    MImax_n_1.append(MImax_n[1])
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plot(tau_range,MImax_n_1,"-o")
xlabel("tau")
ylabel("MI max at n=1")
show()


# In[ ]:


# FULL WIDTH AT HALF MAXIMUM


# In[ ]:


#Spectral radius= 1


# In[125]:


#Spline plotting for n=15
net,X,Y,nrmse_i,mi_i,filename, MImax_n, FWHM=test("Dataset1",file, "noise", 1, [15], nrmse= False, noise=True,gaussian=True)
print(FWHM)


# In[ ]:


#Spectral radius= 0.5


# In[75]:


#Spline plotting for n=15
net,X,Y,nrmse_i,mi_i,filename, MImax_n, FWHM=test("Dataset1",file, "noise", 2, [15], nrmse= False, noise=True,gaussian=True)


# In[ ]:


#range SR


# In[123]:


rho_range=list(np.arange(0.0, 2.1, 0.1))
FWHM_range=[]
for rho in np.arange(0.0, 2.1, 0.1):
    net,X,Y,nrmse_i,mi_i,filename, MImax_n, FWHM=test("Dataset1",file, "noise", rho, [15], nrmse= False, noise=True,gaussian=True)
    FWHM_range.append(FWHM)
plot(rho_range,FWHM_range, "-o")
xlabel("rho")
ylabel("FWHM")
show()


# In[78]:


print(FWHM)


# In[ ]:


## WHITE NOISE
## Parameters
tau=0.01
c_n=1


# In[ ]:


#Mi
net,X,Y,nrmse_i,mi_i,filename, MImax_n_tau_100=test("Dataset1",file, "noise", 0.95, n_range, nrmse= False, noise=True)
exponential_fitting(MImax_n_tau_100, MI=True)


# In[ ]:


#ALL FILES


# In[ ]:


#Nrmse
test_all("Dataset1","noise", 0.95, n_range, nrmse=True, noise=True, euler=True, save=False)


# In[ ]:


#MI
test_all("Dataset1","noise", 0.95, n_range, nrmse=False, noise=True)


# In[ ]:


#                             RESULTS  ROSSLER                                          #


# In[ ]:


# SINGLE FILE:
file=csv_files[0]

## N range
n_range= list(range(10))
i_max=80


# In[ ]:


## Parameters
a=0.15
b=0.20
c=10


# In[ ]:


# EULER


# In[ ]:


## Nrmse
X,Y,nrmse_i,mi_i,filename=test("Dataset1",file, "rossler", 0.95, n_range, nrmse=True, noise=False, euler=True)


# In[ ]:


## Mi
X,Y,nrmse_i,mi_i,filename,MImax_n=test("Dataset1",file, "rossler", 0.95, n_range, nrmse=False, noise=False, euler=True)


# In[ ]:


plot(MImax_n.keys(), MImax_n.values())
show()


# In[ ]:


##ALL FILES
#Nrmse
test_all("Dataset1","rossler", 0.95, n_range, nrmse=True, noise=False, euler=True, save=False)


# In[ ]:


##ALL FILES
#Mi
test_all("Dataset1","rossler", 0.95, n_range, nrmse=False, noise=False, euler=True, save=False)


# In[ ]:


# ODEINT


# In[ ]:


## Nrmse
X,Y,nrmse_i,mi_i,filename=test("Dataset1",file, "rossler", 0.95, n_range, nrmse=True, noise=False, euler= False)


# In[ ]:


## Mi
X,Y,nrmse_i,mi_i,filename=test("Dataset1",file, "rossler", 0.95, n_range, nrmse=False, noise=False, euler= False)


# In[ ]:


##ALL FILES
#Nrmse
test_all("Dataset1","rossler", 0.95, n_range, nrmse=True, noise=False, euler=False, save=False)


# In[ ]:


##ALL FILES
#Mi
test_all("Dataset1","rossler", 0.95, n_range, nrmse=False, noise=False, euler=False, save=False)

