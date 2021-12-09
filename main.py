"""
Main classes.
"""

import numpy as np
from critic_net import assoc_net
from state_net import place_net
from time import time
import os
from pathlib import Path

# File directory
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'

# Main class for actor-critic neural network

class ActorCriticNN:
    
    def __init__(self,params):
        
        # Constants
        self.dt = params['dt']
        self.n_trial = int(params['n_trial'])
        self.v = params['v']
        self.every_perc = params['every_perc']
        self.est_every = params['est_every']
        self.x_dim = params['x_dim']
        self.n_time = int(self.x_dim/self.v/self.dt)
        self.train = params['train']
        
        # Initialize subnetworks
        self.CriticNet = assoc_net(params)
        self.StateNet = place_net(params)
        
    
    def simulate(self):
        # Simulation method
        start = time()
        
        # Save average errors across simulation
        batch_size = int(self.every_perc/100*self.n_trial)
        t_sampl = int(100/self.every_perc)
        self.avg_err = np.zeros((t_sampl,self.n_assoc))
        batch_num = 0
        
        # Store network estimates in single trial level
        if self.est_every:
            self.r = np.zeros((self.n_trial,self.n_time,self.n_assoc))
        
        for j in range(self.n_trial):
            
            # Initialize location
            self.x = 0; self.y = 0
            
            # Inputs to the network
            I_ff = np.zeros((self.n_time,self.n_in))
            
            # Store errors after US appears, omitting transduction delays
            err = np.zeros((self.n_time,self.n_assoc))
            
            for i in range(1,self.n_time):
                
                # Update location
                self.x += self.v * self.dt
                
                # Update state
                r_st = self.StateNet(self.x,self.y)
                I_fb = r_st.flatten()
                
                # Feedforward input
                if self.x> .9 * self.x_dim:
                    I_ff = .1
                else:
                    I_ff = 0
                
                # One-step forward dynamics
                self.CriticNet.dynamics(I_ff,I_fb)
                
                # Weight modification
                if self.train:
                    self.CriticNet.learn_rule()
                
                # Save information
                err[i,:] = self.CriticNet.error
                self.r[j,i,:] = self.CriticNet.r
            
            # Obtain average error at the end of every batch of trials
            if (j % batch_size == 0):
                print('{} % of the simulation complete'.format(round(j/self.n_trial*100)))
                err = np.abs(err)
                self.avg_err[batch_num,:] = np.average(err,0)
                print('Average error is {} Hz'.format(round(1000*np.average(err),2)))
                batch_num += 1
        
        # Simulation time
        end = time()
        self.sim_time = round((end-start)/3600,2)
        print("The simulation ran for {} hours".format(self.sim_time))