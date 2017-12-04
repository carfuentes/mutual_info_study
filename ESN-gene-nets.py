
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


# In[2]:


from preprocessing_net import get_cyclic_net
from mutual_info import memory_capacity_n
from nrmse_calc import nrmse, nrmse_n


# In[3]:


#ESN_class.py imports
import networkx as nx
import numpy as np
from sampling import random_sampling_normal_from_range
import random as rand
import scipy
from preprocessing_net import get_cyclic_net


# In[4]:


#NUMBA


# In[5]:


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
                net[source][target]["weight"]= rand.uniform(0,1)
            elif mode== "-":
                net[source][target]["weight"]= rand.uniform(0,-1)
            elif mode== 0:
                net[source][target]["weight"]= rand.uniform(-1,1)
        return nx.to_numpy_matrix(net)
    
    def initialize(self): 
        np.random.seed(42)
        self.Win=np.random.choice([-0.05,0.05], size=(self.res_size,1+self.in_size))
        self.W0 = np.squeeze(np.asarray(self.W0))
        rhoW0 = max(abs(scipy.linalg.eig(self.W0)[0]))
        self.W= (self.spectral_radius/rhoW0)*self.W0
        

    def collect_states(self, data, init_len, train_len, a=0.3):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        for t in range(train_len):
            u = data[t]
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            if t >= init_len:
                self.X[:,t-init_len]= np.vstack((1,u,self.x))[:,0]
        
        return self.X
    
    
    def collect_states_derivative(self, a,b,c, init_len, train_len, test_len):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t=np.arange(train_len+test_len)
        uyz_x=scipy.integrate.odeint(self.dx_dt,self.x0,t,args=(a,b,c,self.decay))
        self.u=uyz_x[:,0]**2
        self.x_act=uyz_x[:,3:]
        for t in range(init_len,train_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            self.X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return self.X
    
    def dx_dt(self, uyz_x,t,a,b,c,decay):
        u=uyz_x[0]
        y=uyz_x[1]
        z=uyz_x[2]
        x=np.array(uyz_x[3:]).reshape(self.res_size,1)
       
        du_dt=-z-y
        dy_dt=u+a*y
        dz_dt=b+z*(u-c)
        dx_dt=0.5*(np.tanh( np.dot( self.Win, np.vstack((1,u**2)) ) + np.dot( self.W, x ) )+1) - (decay * x)
        return np.insert(dx_dt,0,[du_dt,dy_dt,dz_dt])
    
    def du_dt_rossler(self,z,y):
        return -z-y
    
    def dy_dt_rossler(self, u, a, y):
        return u+a*y
    
    def dz_dt_rossler(self, b, z, u, c):
        return b+z*(u-c)
    
    def dx_dt_euler(self, x,u):
        x=x.reshape(self.res_size,1)
        x_act=self.decay*0.5*(np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, x ) )+1) - (self.decay * x)
        return x_act.reshape(self.res_size)
    
    def colored_noise_euler_integration(self, x_0, u_0, tau, c, t_stop, dt=0.01):
        mu=np.exp(-dt/tau)
        sigma= sqrt( ((c * tau)/2) * (1-mu**2) )
    
    
        t = np.linspace(0, t_stop, int(t_stop/dt))
    
        x=np.zeros((len(t),self.res_size))
        x[0,:]=x_0

        u = u_0 * np.ones_like(t)
    
        for i in range(0, len(t) - 1):
            u[i+1] = u[i]* mu + sigma * np.random.normal()
            x[i+1,:] = x[i,:] + dt * self.dx_dt_euler(x[i,:], u[i])
            
        return u,x
    
    def rossler_euler_integration(self, x_0, u_0, a, b, c, t_stop, dt=0.01):
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
            x[i+1,:] = x[i,:] + dt * self.dx_dt_euler(x[i,:], u[i]**2)
            
        return u,x

    def collect_states_euler(self, tau, c, init_len, train_len, test_len,dt=0.001):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t_stop=train_len+test_len
        u, x_act=self.colored_noise_euler_integration(self.x0_e, self.u0, tau, c, t_stop, dt)

        
        indexes=[int(t/dt) for t in range(0,t_stop)]
        self.u=u[indexes]
        self.x_act=x_act[indexes]
   
     
        for t in range(init_len,train_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            self.X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return self.X
    
     
    def collect_states_euler_rossler(self, a, b, c, init_len, train_len, test_len,dt=0.01):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t_stop=train_len+test_len
        u, x_act=self.rossler_euler_integration(self.x0_e, self.u0, a,b, c, t_stop, dt)

        
        indexes=[int(t/dt) for t in range(0,t_stop)]
        self.u=u[indexes]**2
        self.x_act=x_act[indexes]
     
        for t in range(init_len,train_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            self.X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return self.X
        
        
    def calculate_weights(self, data, init_len, train_len,beta=1e-8 ):
        Y=data[None,init_len+1:train_len+1]
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1)))
        return self.Wout
    
    def calculate_weights_derivative(self,init_len, train_len, n, beta=1e-8 ):
        Y=np.array([self.u[init_len-n:train_len-n]])
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1))) #w= y*x_t*(x*x_t + beta*I)^-1
        return self.Wout
    
    
    def run_generative(self, data, test_len, train_len,a=0.3):
        self.Y = np.zeros((self.out_size,test_len))
        u = data[train_len]
        for t in range(test_len):
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
            u = data[trainLen+t+1]
            #u =y
    
    def run_predictive_derivative(self, test_len, train_len):
        self.Y = np.zeros((self.out_size,test_len))
        
        for t in range(train_len,train_len+test_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            y = np.dot( self.Wout, np.vstack((1,u_concat,x_concat)) )
            self.Y[:,t-train_len] = y
           
        
        return self.Y


# In[6]:


def colored_noise_euler_integration(self, x_0, u_0, tau, c, t_stop, dt=0.01):
    mu=np.exp(-dt/tau)
    sigma= sqrt( ((c * tau)/2) * (1-mu**2) )


    t = np.linspace(0, t_stop, int(t_stop/dt))

    x=np.zeros((len(t),self.res_size))
    x[0,:]=x_0

    u = u_0 * np.ones_like(t)

    for i in range(0, len(t) - 1):
        u[i+1] = u[i]* mu + sigma * np.random.normal()
        x[i+1,:] = x[i,:] + dt * self.dx_dt_euler(x[i,:], u[i])
        
    return u,x


# In[7]:


##################################################################################


# In[8]:


#                                   FUNCTIONS                                    #


# In[9]:


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


# In[10]:


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


# In[11]:


def testing_gene_net_derivative(directory,a,b,c,n,i_max=80):
    #init
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    Y_by_file={}
    X_by_file={}
    MI_by_file={}
    NRMSE_by_file={}
    
    for file in csv_files:
        #init
        print(file)
        filename=file[file.index("list")+5:file.index(".csv")]
        
        #Run network
        net=ESN(os.path.join(directory, file),1,1,0.95)
        net.initialize()
        net.collect_states_derivative(a,b,c,initLen,trainLen,testLen)
        net.calculate_weights_derivative(initLen,trainLen,n)
        net.run_predictive_derivative(testLen,trainLen)
        
        #Calculate output
        X_by_file[filename]=net.u
        Y_by_file[filename]=net.Y
        MI_by_file[filename]=memory_capacity_n(net.Y, net.u,n)
        NRMSE_by_file[filename]=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen)
        
        #prints
        
        print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
        print(net.res_size, " FINISHED")
    return X_by_file, Y_by_file, NRMSE_by_file, MI_by_file
  


# In[12]:


def testing_gene_net_derivative_file(directory,file,a,b,c,n,i_max=80):
    #init
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    
    #Run network
    net=ESN(os.path.join(directory, file),1,1,0.95)
    net.initialize()
    net.collect_states_derivative(a,b,c,initLen,trainLen,testLen)
    net.calculate_weights_derivative(initLen,trainLen,n)
    net.run_predictive_derivative(testLen,trainLen)
    
    #Calculate output
    X=net.u
    Y=net.Y
    mi_i=memory_capacity_n(net.Y, net.u,n)
    nrmse_i=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen)
    
    #prints
    print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
    print(net.res_size, " FINISHED")
    
    return X,Y,nrmse_i,mi_i


# In[13]:


def testing_gene_net_euler_file(directory,file,tau,c,n,i_max=80):
    #init
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    
    #Run network
    net=ESN(os.path.join(directory, file),1,1,0.95)
    net.initialize()
    net.collect_states_euler(tau,c,initLen,trainLen,testLen,dt=0.001)
    net.calculate_weights_derivative(initLen,trainLen,n)
    net.run_predictive_derivative(testLen,trainLen)
    
    #Calculate output
    X=net.u
    Y=net.Y
    mi_i=memory_capacity_n(net.Y, net.u,n)
    nrmse_i=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen)
    
    #prints
    print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
    print(net.res_size, " FINISHED")
    
    return X,Y,nrmse_i,mi_i


# In[14]:


def testing_gene_net_euler_rossler_file(directory,file,a,b,c,n,i_max=80):
    #init
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    
    #Run network
    net=ESN(os.path.join(directory, file),1,1,0.95)
    net.initialize()
    net.collect_states_euler_rossler(a,b,c,initLen,trainLen,testLen,dt=0.01)
    net.calculate_weights_derivative(initLen,trainLen,n)
    net.run_predictive_derivative(testLen,trainLen)
    
    #Calculate output
    X=net.u
    Y=net.Y
    mi_i=memory_capacity_n(net.Y, net.u,n)
    nrmse_i=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen)
    
    #prints
    print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
    print(net.res_size, " FINISHED")
    
    return X,Y,nrmse_i,mi_i


# In[15]:


def testing_gene_net_euler(directory,tau,c,n,i_max=80):
    #init
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    Y_by_file={}
    X_by_file={}
    MI_by_file={}
    NRMSE_by_file={}
    
    for file in csv_files:
        #init
        print(file)
        filename=file[file.index("list")+5:file.index(".csv")]
        
        #Run network
        net=ESN(os.path.join(directory, file),1,1,0.95)
        net.initialize()
        net.collect_states_euler(tau,c,initLen,trainLen,testLen,dt=0.01)
        net.calculate_weights_derivative(initLen,trainLen,n)
        net.run_predictive_derivative(testLen,trainLen)
        
        #Calculate output
        X_by_file[filename]=net.u
        Y_by_file[filename]=net.Y
        MI_by_file[filename]=memory_capacity_n(net.Y, net.u,n)
        NRMSE_by_file[filename]=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen)
        
        #prints
        
        print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
        print(net.res_size, " FINISHED")
    return X_by_file, Y_by_file, NRMSE_by_file, MI_by_file
  


# In[16]:


array=np.zeros((18,trainLen-initLen))
array.shape
array[:,8799]


# In[20]:


def testing_gene_net_euler_rossler(directory,a,b,c,n,i_max=80):
    #init
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    Y_by_file={}
    X_by_file={}
    MI_by_file={}
    NRMSE_by_file={}
    
    for file in csv_files:
        #init
        print(file)
        filename=file[file.index("list")+5:file.index(".csv")]
        
        #Run network
        net=ESN(os.path.join(directory, file),1,1,0.95)
        net.initialize()
        net.collect_states_euler_rossler(a,b,c,initLen,trainLen,testLen,dt=0.01)
        net.calculate_weights_derivative(initLen,trainLen,n)
        net.run_predictive_derivative(testLen,trainLen)
        
        #Calculate output
        X_by_file[filename]=net.u
        Y_by_file[filename]=net.Y
        MI_by_file[filename]=memory_capacity_n(net.Y, net.u,n)
        NRMSE_by_file[filename]=nrmse_n(net.Y,net.u,i_max,errorLen,trainLen)
        
        #prints
        
        print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
        print(net.res_size, " FINISHED")
    return X_by_file, Y_by_file, NRMSE_by_file, MI_by_file
  


# In[21]:


def plot_dict_i(key,dict_i,nrmse=True, single=True):
    x=[]
    y=[]
    
    if nrmse:
        point=min(dict_i, key=dict_i.get)
        print(key,"=> min=",point, "value", dict_i[point])
    else:
        point=max(dict_i, key=dict_i.get)
        print(key,"=> max=",point)
        
    for i,value in dict_i.items():
        x.append(i)
        y.append(value)
    
    plot(x,y,label=key)
    plot(point,dict_i[point],marker='o')
    
    if single:
        xlabel("i=[0,%d]"%(x[-1]))
        
        if nrmse:
            ylabel("NRMSE(X(t-i), Y(t))")
        else:
            ylabel("MI(X(t-i), Y(t))")


# In[22]:


def plot_dict_by_file(dict_by_file,n,nrmse=True,save=True):
    for file in dict_by_file.keys():
        plot_dict_i(file, dict_by_file[file],nrmse=nrmse,single=False)
    legend(loc='upper left')
    xlabel("i=[0,%d]"%(len(dict_by_file[file].keys())-1))
    
    if nrmse:
        ylabel("NRMSE(X(t-i), Y(t))")
    else:
        ylabel("MI(X(t-i), Y(t))")
        
    if save:
        savefig("plots/nrmse_i_all_files/nrmse_all_n%d" %(n))
    show()
    


# In[41]:


def plot_temporal_lines(u,Y,n,length,filename, save=True):
    plot( u[trainLen-n:trainLen+length-n], 'g' )
    plot( Y.T[0:length], 'b' )
    xlabel("time")
    ylabel("signal")
    title('Target and generated signals $y(t)$ starting at $t=0$ until $t=%s$ with a delay of n=%d ' %(length,n))
    legend(['Target signal', 'Free-running predicted signal'])
    if save:
        savefig("plots/input-vs-output/%s_len%d_n%d_decay_random" %(filename,length,n))
    show()


# In[24]:


def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


# In[25]:


##################################################################################


# In[26]:


#                                  PARAMETERS                                    #


# In[27]:


# TRAINING AND TEST LENGHT
errorLen = 500
trainLen=9000
testLen=1000
initLen=200

#Files
csv_files=['network_edge_list_ENCODE.csv', 'network_edge_list_modENCODE.csv', 'network_edge_list_YEASTRACT.csv', 'network_edge_list_EcoCyc.csv', 'network_edge_list_DBTBS.csv']

#parameters ROSSLER
a=0.1
b=0.1
c_r=14

#parameters COLORED NOISE
tau = 100
c=10


# In[28]:


##################################################################################


# In[29]:


#                                   TESTEOS                                      #


# In[30]:


# TESTEO get_cyclic_net
G=get_cyclic_net(os.path.join("Dataset1/", "network_edge_list_modENCODE.csv"))
len(G.nodes())


# In[31]:


#TESTEO adjacency matrix
net=ESN(os.path.join("Dataset1/", "network_edge_list_DBTBS.csv"),1,1,0.95)
net.W0


# In[32]:


#TESTEO initialize
net.initialize()
print(net.W.shape)
print(max(abs(scipy.linalg.eig(net.W)[0])))


# In[33]:


#TESTEO collect states
net.collect_states_derivative(a,b,c, initLen, trainLen, testLen)
net.X.shape
net.X[:,7]


# In[34]:


##################################################################################


# In[35]:


#                             RESULTS  NOISE                                         #


# In[36]:


#una n


# In[ ]:


X,Y,nrmse_i,mi_i=testing_gene_net_euler_file("Dataset1",csv_files[-1],tau=tau,c=c,n=0)
plot_dict_i("DBTBS", nrmse_i)
show()


# In[ ]:


plot(Y.T[0:100],"g")
show()
print(Y.T)


# In[ ]:


plot_dict_i("DBTBS", nrmse_i)
show()


# In[ ]:


# rango de n


# In[42]:


file=csv_files[3]
filename=file[file.index("list")+5:file.index(".csv")]

for n in [0,1,2,5,10]:
    X,Y,nrmse_i,mi_i=testing_gene_net_euler_file("Dataset1",file,tau=200,c=1,n=n)
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_i(filename, nrmse_i)
    #savefig("plots/nrmse_i/%s_n%d_decay_random" %(filename,n))
    show()
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n,testLen,filename, save=False)
    
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, 50,filename, save=False)


# In[ ]:


print(Y)


# In[ ]:


file=csv_files[-1]
filename=file[file.index("list")+5:file.index(".csv")]
n=2

for tau in range(1,11):
    X,Y,nrmse_i,mi_i=testing_gene_net_euler_file("Dataset1",file,tau=tau,c=c,n=n)
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("tau="+str(tau))
    plot_dict_i(filename, nrmse_i)
    #savefig("plots/nrmse_i/%s_n%d_decay_random" %(filename,n))
    show()
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, testLen,filename, save=False)
    
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, 50,filename, save=False)


# In[46]:


for n in range(10):
    X_by_file, Y_by_file, NRMSE_by_file,MI_by_file=testing_gene_net_euler("Dataset1/", tau,c,n)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_by_file(NRMSE_by_file,n,save=False)


# In[ ]:


#                             RESULTS  ROSSLER                                          #


# In[ ]:


##ALL FILES


# In[ ]:


print("With derivatives")
X_by_file, Y_by_file, NRMSE_by_file, MI_by_file=testing_gene_net_derivative("Dataset1/", a,b,c,0)


# In[ ]:


for n in [0,1,2,5,10,25]:
    X_by_file, Y_by_file, NRMSE_by_file,MI_by_file=testing_gene_net_derivative("Dataset1/", a,b,c,n)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_by_file(NRMSE_by_file,n,save=False)


# In[ ]:


##SINGLE FILE
#una n


# In[ ]:


X,Y,nrmse_i,mi_i=testing_gene_net_derivative_file("Dataset1",csv_files[-1],a=a,b=b,c=c,n=20)
plot_dict_i("DBTBS", nrmse_i)
show()


# In[ ]:


#rango de n


# In[43]:


file=csv_files[2]
filename=file[file.index("list")+5:file.index(".csv")]

for n in [0,1,2,3,25,30]:
    X,Y,nrmse_i,mi_i=testing_gene_net_derivative_file("Dataset1",file,a=a,b=b,c=c,n=n)
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_i(filename, nrmse_i)
    savefig("plots/nrmse_i/%s_n%d_decay_random" %(filename,n))
    show()
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, testLen,filename)
    
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, 50,filename)
   


# In[44]:


file=csv_files[3]
filename=file[file.index("list")+5:file.index(".csv")]

for n in [0,1,2,5,10,20]:
    X,Y,nrmse_i,mi_i=testing_gene_net_euler_rossler_file("Dataset1",file,a,b,c,n=n)
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_i(filename, nrmse_i)
    #savefig("plots/nrmse_i/%s_n%d_decay_random" %(filename,n))
    show()
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, testLen,filename, save=False)
    
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, 50,filename, save=False)
   


# In[45]:


for n in [1,0,2,5,10,25]:
    X_by_file, Y_by_file, NRMSE_by_file,MI_by_file=testing_gene_net_euler_rossler("Dataset1/", a,b,c,n)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_by_file(NRMSE_by_file,n,save=False)


# In[ ]:


#MUTUAL INFO
file=csv_files[-1]
filename=file[file.index("list")+5:file.index(".csv")]

for n in [0,15,20,25,30,50,60,80]:
    X,Y,nrmse_i,mi_i=testing_gene_net_derivative_file("Dataset1",file,a=a,b=b,c=c,n=n)
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_dict_i(filename, mi_i,nrmse=False)
    #savefig("plots/nrmse_i/%s_n%d" %(filename,n))
    show()
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, testLen,filename, save=False)
    
    
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot_temporal_lines(X,Y, n, 50,filename,save=False)
   

