#imports
import numpy as np
from statistics import mean



#functions
def random_sampling_normal_from_range(list_range,size):
    m= mean(list_range)
    s= abs((list_range[1] - m)/3)
    return np.random.normal(m,s,size)



