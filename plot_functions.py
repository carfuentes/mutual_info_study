#imports
from matplotlib.pyplot import *
from parameters import startLen
from graph_analysis import spline_plotting
from matplotlib import rc
from matplotlib import colors as mcolors
from parameters import *
import string


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
rc('font',**{'family':'serif','serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#functions

def plot_dict_i(key,dict_i,MImax_n,n,index,ax,max_value,nrmse=True):
    x=[]
    y=[]
    
    if nrmse:
        name='dodgerblue'
        print("Plot nrmse")
        point=min(dict_i, key=dict_i.get)
        print(key,"=> min=",point, "value", dict_i[point])
    else:
        name="teal"
        print("Plot mi")
        point=max(dict_i, key=dict_i.get)
        print(key,"=> max=",point, "value", dict_i[point])
        MImax_n[n]=dict_i[point]
        
    for i,value in dict_i.items():
        x.append(i)
        y.append(value)
  
        ax.plot(y,"-o",color=colors[name])
        #h=ax.plot(y,"-o")
        ax.tick_params(axis='both', labelsize=22)
        ax.set_title("n="+str(n),fontsize=30)
        ax.text(-0.19, 1.1, string.ascii_lowercase[index]+")", transform=ax.transAxes, 
            size=30)
             
        #plot(point,dict_i[point],marker='o')
        #annotate("{:.2}".format(dict_i[point]),xy=(point,dict_i[point]),xytext=(point+0.01,dict_i[point]+0.01),fontsize=14)
        
        #if index==50:
        #    ax1.set_xlabel("i",fontsize=30)
        #return h
        

def plot_nrmse_mi_by_n(n_list,nrmse_by_n,MImax_n,mi_by_n):
    fig, axes = subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize=(20, 10))
    #subplots_adjust(wspace=0.3,hspace=0.5)
    fig.text(0.5, 0.04, 'time lags (i)', ha='center',fontsize=30)
    for i,n in enumerate(n_list):
        if i==0:
            axes[0,i].text(-0.4, 0.5, 'NRMSE', va='center',rotation="vertical",fontsize=30,transform=axes[0,i].transAxes)
            axes[1,i].text(-0.4, 0.5, 'MI', va='center',rotation="vertical",fontsize=30,transform=axes[1,i].transAxes)
            #axes[0,i].text(-0.5, 1.2, '1.', va='center',fontsize=30,fontweight="heavy",transform=axes[0,i].transAxes)
            #axes[1,i].text(-0.5, 1.2, '2.', va='center',fontsize=30,fontweight="heavy",transform=axes[1,i].transAxes)
        axes[0,i].set_ylim(bottom=0,top=1.6)
        plot_dict_i("key",nrmse_by_n[n],MImax_n,n,i,axes[0,i],1.6,nrmse=True)
        axes[1,i].set_ylim(bottom=0,top=0.35)
        plot_dict_i("key",mi_by_n[n],MImax_n,n,i+4,axes[1,i],0.2,nrmse=False)
    show()

    

def plot_temporal_lines(u,ax2,Y,n,length,filename, tau,folder,save=True):
    #plot( u[startLen-n:startLen+length-n], 'g' )
    #name=["crimson","royalblue","seagreen"]
    #rc('text', usetex=True)
    #rc('font', family='serif')
    ax2.plot( u[trainLen-n:trainLen+length-n], color=colors['dodgerblue'],linewidth=5 )
    ax2.plot( Y.T[0:length],"--", color=colors['crimson'],linewidth=3 )
    ax2.tick_params(axis='both', labelsize=20)
    ax2.text(-0.1, 1.1, "b)", transform=ax2.transAxes, size=20)
    #xlabel("time",fontsize=40
    #ylabel("signal",fontsize=40)
    #title('Target and generated signals $y(t)$ starting at $t=0$ until $t=%s$ with a delay of n=%d ' %(length,n),fontsize=16)
    ax2.legend(['Noise signal with delay n= {}'.format(n), "Predicted signal"],fontsize=18)
    if save:
        savefig("plots/input-vs-output/%s/%s_len%d_n%d_tau%s" %(folder,filename,length,n,str(tau)))
    show()


def plot_dict_i_FWHM(key,dict_i,MImax_n,n,nrmse=True, single=True, gaussian=False):
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
    
  
    plot(y,"-o",color=colors["teal"])
    tick_params(axis='both', labelsize=20)
    title("n="+str(n),fontsize=26)
    
    
    if single:
        xlabel("time lags (i)",fontsize=26)
        
        if nrmse:
            ylabel("NRMSE(X(t-i), Y(t))")
        else:
            ylabel("Mutual Information",fontsize=26)
    if gaussian:
        FWHM = spline_plotting(x,y,nrmse)
    
    return FWHM

def get_figure_noise(net,Y_n):
    fig, axes = subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(10, 10))
    fig.text(0.5, 0.04, 'time', ha='center',fontsize=30)

    #subplots_adjust(wspace=0.3)
    ax1=axes[0]
    ax2=axes[1]
    ax1.text(-0.1, 1.1, "a)", transform=ax1.transAxes, size=20)       
    n=10

    ax1.plot( net.u[trainLen-n:trainLen+100-n], color=colors['dodgerblue'],linewidth=5 )
    ax1.plot( net.u[trainLen:trainLen+100], "--",color=colors['grey'],linewidth=2,alpha=0.5)
    ax1.legend(['Noise signal with delay n= {}'.format(n),"Noise signal"],fontsize=18)
    ax1.tick_params(axis='both', labelsize=20)

    plot_temporal_lines(net.u,ax2,Y_n[10], 10,100,"encode", tau, "none", False)

    ax2.tick_params(axis='both', labelsize=20)