#imports
import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brute

#functions
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    result=result[int(result.size/2):]
    return result/result[0]

def func(x, a, b, c):
    return a * np.exp(float(-1/b) * x) + c

def exponential_fitting(x,MI=False):
    if MI: #x is MI
        xdata=np.array(list(x.keys())[1:])
        ydata=np.array(list(x.values())[1:])
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
    return popt

def spline_plotting(data,notebook=False):
    x=list(data.keys())
    y=list(data.values())
    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM=abs(r1-r2)
    print("r1=%f and r2=%f"%(r1,r2))
    print("FWHM",FWHM)

    if notebook:
        plot(x, y)
        plot(r1,0,marker='o')
        plot(r2,0,marker='o')
        axvspan(r1, r2, facecolor='g', alpha=0.5)
        show()
    
    return FWHM