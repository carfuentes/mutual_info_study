#imports
import networkx as nx
import numpy as np
import random as rand
import scipy
from preprocessing_net import get_cyclic_net
from math import sqrt


from parameters import tau, c_n

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
    
    def initialize(self,i_scaling,beta_scaling): 
        np.random.seed(42)
        print("factor i= %f"%i_scaling)
        print("factor beta= %f"%beta_scaling)
        self.Win=np.random.uniform(size=(self.res_size,1+self.in_size))*i_scaling
        self.Win[:,0]=np.random.uniform(size=(self.res_size,))*beta_scaling
        self.W0 = np.squeeze(np.asarray(self.W0))
        radius = np.max(np.abs(np.linalg.eigvals(self.W0)))
        self.W= (self.spectral_radius/radius)*self.W0
        return self.W
    
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
        x=x.reshape(self.res_size,1)
        x_act=self.decay*0.5*(np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, x ) )+1) - (self.decay * x)
        return x_act.reshape(self.res_size)
    
    def colored_noise_euler_integration(self, x_0, u_0, tau, c, t_stop, dt=0.001):
        mu=np.exp(-dt/tau)
        sigma= sqrt( ((c * tau)/2) * (1-mu**2) )
    
    
        t = np.linspace(0, t_stop, int(t_stop/dt))
    
        x=np.zeros((len(t),self.res_size))
        x[0,:]=x_0

        u = u_0 * np.ones_like(t)
    
        for i in range(0, len(t) - 1):
            u[i+1] = u[i]* mu + sigma * np.random.normal()
            x[i+1,:] = x[i,:] + dt * self.dx_act_dt(x[i,:], u[i])
            
        return u,x
    
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

    def collect_states_derivative_OLD(self, init_len, train_len, test_len, euler, noise,dt=0.001):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t_stop=train_len+test_len
        
        if not euler:
            print("Collecting states with rossler input using odeint built in...")
            print("Parameters: a= %.2f b= %.2f c= %.2f"%(a,b,c))
            t=np.arange(train_len+test_len)
            uyz_x=scipy.integrate.odeint(self.dx_dt,self.x0,t,args=(a,b,c,self.decay))
            self.u=uyz_x[:,0]
            self.x_act=uyz_x[:,3:]
            
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
            self.u=u[indexes]
            self.x_act=x_act[indexes]
   
        for t in range(init_len,train_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            self.X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return self.X 

    def calculate_weights_derivative_OLD(self,init_len, train_len, n, beta=1e-8 ):
        Y=np.array([self.u[init_len-n:train_len-n]])
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1))) #w= y*x_t*(x*x_t + beta*I)^-1
        return self.Wout
    
    def run_predictive_derivative_OLD(self, test_len, train_len):
        self.Y = np.zeros((self.out_size,test_len))
        for t in range(train_len,train_len+test_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            y = np.dot( self.Wout, np.vstack((1,u_concat,x_concat)) )
            self.Y[:,t-train_len] = y 
        
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
