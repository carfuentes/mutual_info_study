def collect_states_derivative(self, init_len, train_len, test_len, euler, noise,dt=0.001):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t_stop=train_len+test_len
        
        if not euler:
            print("Collecting states with rossler input using odeint built in...")
            print("Parameters: a= %.2f b= %.2f c= %.2f"%(a,b,c))
            t=np.arange(train_len+test_len)
            uyz_x=scipy.integrate.odeint(self.dx_dt,self.x0,t,args=(a,b,c,self.decay))
            self.u=uyz_x[:,0]**2
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
            self.u=u[indexes]**2
            self.x_act=x_act[indexes]
   
        for t in range(init_len,train_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            self.X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return self.X
