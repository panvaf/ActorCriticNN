"""
State network.
"""

import numpy as np


# Place cell network class


class place_net:
    
    def __init__(self,params):
        # Initialize network of place cells
        
        # Constants
        self.sigma = params['sigma']
        self.s = params['s']
        self.x_dim = params['x_dim']
        self.y_dim = params['y_dim']
        self.neu_den = params['neu_den']
        
        self.n_x = int(self.x_dim * self.neu_den)
        if self.y_dim is not None: 
            self.n_y = int(self.y_dim * self.neu_den)
        
        # Place cell locations
        if self.y_dim is not None:
            x0 = np.linspace(0,self.x_dim,self.n_x)
            y0 = np.linspace(0,self.y_dim,self.n_y)
            self.x0, self.y0 = np.meshgrid(x0,y0)
        else:
            self.x0 = np.linspace(0,self.x_dim,self.n_x)
            self.y0 = 0
            
    
    
    def fr(self,x,y):
        # Returns firing rates from x,y location
        
        d_sq = (x-self.x0)**2 + (y-self.y0)**2
        self.r = self.s * np.exp(-d_sq/(2*self.sigma))
        
        return self.r