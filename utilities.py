"""
Utilities.
"""

import numpy as np

# Choose activation function

def act_fun(x,fun):
    
    if fun == 'rect':
        r = rect(x)
    elif fun == 'logistic':
        r = logistic(x)
    
    return r


# Logistic function
    
def logistic(x,x0=1.5,b=2,s=.1):
    # s: maximum firing rate in kHz
    # x0: 50 % firing rate point
    # b: steepness of gain
    
    return s/(1+np.exp(-b*(x-x0)))


# Rectification function

def rect(x,s=.1):
    x[x<0] = 0
    x[x>1] = 1
    return s*x


# Place cell class

class place_cell:
    
    def __init__(self,params):
        # Initialize place cell
        
        self.x0 = params['x0']
        self.y0 = params['y0']
        self.sigma = params['sigma']
        self.s = params['s']
        
    
    def fr(self,x,y):
        
        d_sq = (x-self.x0)**2 + (y-self.y0)**2
        self.r = self.s * np.exp(-d_sq/(2*self.sigma))
        
        return self.r