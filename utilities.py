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
    
    
# Returns filename from network parameters

def filename(params):
    
    filename =  format(params['n_trial'],'.0e').replace('+0','') + 'trials' + \
        (('insz' + str(params['n_in'])) if params['n_in'] != 20 else '') + \
        (('PlCell' + str(params['n_pl'])) if params['n_pl'] != 64 else '') + \
        (('taus' + str(params['tau_s'])) if params['tau_s'] != 10 else '') + \
        (('taulp' + str(params['tau_lp'])) if params['tau_lp'] is not None else '') + \
        'd' + str(params['d']) + \
        (('n' + str(params['n_sigma']).replace(".","")) if params['n_sigma'] != 0 else '') + \
        (('N' + str(params['n_assoc'])) if params['n_assoc'] != 64 else '') + \
        (('eta' + str(params['eta'])) if params['eta'] != 1e-2 else '') + \
        ('Dale' if params['dale'] else '')
        
    return filename