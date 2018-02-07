#imports
from matplotlib.pyplot import *
from parameters import startLen
from graph_analysis import spline_plotting

#functions

def plot_dict_i(key,dict_i,MImax_n,n,nrmse=True, single=True, notebook=False):
    x=[]
    y=[]
    
    if nrmse:
        print("Plot nrmse")
        point=min(dict_i, key=dict_i.get)
        print(key,"=> min=",point, "value", dict_i[point])
    else:
        print("Plot mi")
        point=max(dict_i, key=lambda y: abs(dict_i[y]))
        print(key,"=> max=",point)
        MImax_n[n]=abs(dict_i[point])
        
    for i,value in dict_i.items():
        x.append(i)
        y.append(value)
    
    if notebook:
        figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        title("n="+str(n))
        plot(x,y,"-o",label=key)
        plot(point,dict_i[point],marker='o')
    
    
        if single:
            xlabel("i=[0,%d]"%(x[-1]))
        
            if nrmse:
                ylabel("NRMSE(X(t-i), Y(t))")
            else:
                ylabel("MI(X(t-i), Y(t))")



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

    

def plot_temporal_lines(u,Y,n,length,filename, tau,folder,save=True):
    plot( u[startLen-n:startLen+length-n], 'g' )
    plot( Y.T[0:length], 'b' )
    xlabel("time")
    ylabel("signal")
    title('Target and generated signals $y(t)$ starting at $t=0$ until $t=%s$ with a delay of n=%d ' %(length,n))
    legend(['Target signal', 'Free-running predicted signal'])
    if save:
        savefig("plots/input-vs-output/%s/%s_len%d_n%d_tau%s" %(folder,filename,length,n,str(tau)))
   