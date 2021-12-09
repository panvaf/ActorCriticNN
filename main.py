"""
Main classes.
"""

import numpy as np
import utilities as util
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
        
        # Initialize subnetworks
        self.CriticNet = assoc_net(params)
        self.StateNet = place_net(params)

    
    def simulate(self):
        # Simulation method
        start = time()
        
        n_time = int(params['x_dim']/self.v/self.dt)
        
        # Save average errors across simulation
        batch_size = int(self.every_perc/100*self.n_trial)
        t_sampl = int(100/self.every_perc)
        self.avg_err = np.zeros((t_sampl,self.n_assoc))
        batch_num = 0
        
        # Store network estimates in single trial level
        if self.est_every:
            self.r = np.zeros(tuple([self.n_trial])+self.CS.shape)
            self.Phi_est = np.zeros(tuple([self.n_trial])+self.Phi.shape)
            self.R_est = np.zeros(tuple([self.n_trial])+self.R.shape)
        
        for j, trial in enumerate(trials):
            
            # Inputs to the network
            I_ff = np.zeros((self.n_time,self.n_in)); I_ff[self.n_US_ap:,:] = self.US[trial,:]
            g_inh = np.zeros(self.n_time); g_inh[self.n_US_ap:] = self.g_inh
            R = np.zeros(self.n_time); R[self.n_US_ap+n_trans] = self.R[trial]
            R_est = 0
            
            I_fb = np.zeros((self.n_time,self.n_fb)); I_fb[0:self.n_CS_disap,:] = self.CS[trial,:]
            
            # Store errors after US appears, omitting transduction delays
            err = np.zeros((self.n_time-self.n_US_ap,self.n_assoc))
            
            for i in range(1,self.n_time):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i  = assoc_net.dynamics(r,
                                I_ff[i,:],I_fb[i,:],self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt_ms,
                                self.n_sigma,g_inh[i],self.I_inh,self.fun,self.tau_s)
                
                # Weight modification
                if self.train:
                    self.W_rec, self.W_fb = assoc_net.learn_rule(self.W_rec,
                                self.W_fb,r,error,Delta,PSP,eta,self.dt_ms,
                                self.dale,self.S,self.filter,self.rule,self.norm)
                    if i>self.n_US_ap+n_trans:
                        err[i-self.n_US_ap-n_trans,:] = error
                        
            # Save network estimates after each trial
            if self.est_every:
                self.US_est[j,:], self.Phi_est[j,:] = self.est_US()
                self.R_est[j,:], _ = self.est_R(self.US_est[j,:])
            
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
        
        # Final network estimates
        if not self.est_every:
            self.US_est, self.Phi_est = self.est_US()
            self.R_est, _ = self.est_R(self.US_est)