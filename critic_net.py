"""
Critic network.
"""

import numpy as np
import utilities as util

# Associative network class

class assoc_net:
    
    def __init__(self,params):
        # Initialize network
        
        # Constants
        self.dt = params['dt']
        self.n_sigma = params['n_sigma']
        self.n_neu = int(params['n_assoc'])
        self.tau_s = params['tau_s']
        self.n_ff = int(params['n_in'])
        self.n_fb = int(params['n_pl'])
        self.eta = params['eta']
        self.dale = params['dale']
        self.rule = params['rule']
        self.tau_lp = params['tau_lp']
        self.d = params['d']
        self.alpha = self.dt/self.tau_s
        
        self.g_inh = 3*np.sqrt(1/self.n_neu)
        self.E_e = 14/3
        self.E_i = -1/3
        
        # Weights
        self.W_rec = np.zeros((self.n_neu,self.n_neu))
        self.W_ff = np.ones((self.n_neu,self.n_ff))
        self.W_fb = np.random.normal(0,np.sqrt(1/self.n_neu),(self.n_assoc,self.n_fb))
        
        # Initial conditions
        self.V_d = np.random.uniform(0,1,self.n_assoc) 
        self.V = np.random.uniform(0,1,self.n_assoc)
        self.I_d = np.zeros(self.n_assoc)
        self.Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_ff))
        self.PSP = np.zeros(self.n_assoc+self.n_ff)
        self.I_PSP = np.zeros(self.n_assoc+self.n_ff)
        self.PSP_lp = np.zeros(self.n_assoc+self.n_ff)
        self.g_e = np.zeros(self.n_assoc)
        self.g_i = np.zeros(self.n_assoc)
        self.r = np.random.uniform(0,.15,self.n_assoc)
        self.V_ss = np.zeros(self.n_assoc)
        self.r_hat = np.zeros(self.n_assoc)
        
    
    
    def dynamics(self,I_ff,I_fb,tau_l=20,gD=.2,gL=.1):
        # Network dynamics
    
        # Constants
        c = gD/gL
        p = gD/(gD+gL)
        
        # Create noise that will be added to all origins of input
        n = np.random.normal(0,self.n_sigma,self.n_neu)
        n_d = np.random.normal(0,self.n_sigma,self.n_neu)
        
        # input to the dendrites
        self.I_d += (- self.I_d + np.dot(self.W_rec,self.r) + np.dot(self.W_fb,I_fb) + n_d)*self.alpha
        
        # Dentritic potential is a low-pass filtered version of the dentritic current
        self.V_d += (-self.V_d+self.I_d)*self.dt/tau_l
        
        # Time-dependent somatic conductances
        self.g_e += (-self.g_e + np.dot(self.W_ff.clip(min=0),I_ff))*self.alpha
        self.g_i += (-self.g_i - np.dot(self.W_ff.clip(max=0),I_ff))*self.alpha
        
        # Input to the soma (teacher signal)
        I = self.g_e*(self.E_e-self.V) + (self.g_i+self.g_sh)*(self.E_i-self.V)
        
        # Somatic voltage
        self.V += (-self.V + c*(self.V_d-self.V) + I/gL + n)*self.dt/tau_l
        
        # Firing rate
        self.r = util.act_fun(self.V,self.fun)
        
        # Dendritic prediction of somatic voltage & firing rate
        self.V_ss = p * self.V_d
        self.r_hat = util.act_fun(self.V_ss,self.fun)
        
        # Discrepancy between dendritic prediction and actual firing rate
        self.error = self.r - self.r_hat
        
        # Compute PSP for every dendritic input to associative neurons
        r_in = np.concatenate((self.r,I_fb))
        self.I_PSP += (- self.I_PSP + r_in)*self.alpha
        self.PSP += (- self.PSP + self.I_PSP)*self.dt/tau_l
        
        # Low pass filter PSP for bootstrapping
        if self.tau_lp is not None:
            self.PSP_lp += (- self.PSP_lp + self.PSP)*self.dt/self.tau_lp
        else:
            self.PSP_lp = self.PSP



    def learn_rule(self,filt=False,tau_d=100):
        # Learning dynamics
        
        # Weight update
        PI = self.d * np.outer(self.r,self.PSP_lp) - np.outer(self.r_hat,self.PSP)
       
        # Low-pass filter weight updates
        if filt:
            self.Delta += (PI - self.Delta)*self.dt/tau_d
        else:
            self.Delta = PI
        dW = self.eta*self.Delta*self.dt
            
        # Separate matrices
        dW_rec, dW_fb = np.split(dW,[self.n_neu],axis=1)
        
        # Perform weight updates
        self.W_rec += dW_rec
        self.W_fb += dW_fb
        
        # Set every weight that violates Dale's law to zero
        if self.dale:
            self.W_rec[self.W_rec<0] = 0



    def ss_fr(self,I_ff):
        # Find instructed steady-state firing rate for given feedforward input
        
        # Steady-state somatic conductances
        g_e = np.dot(self.W_ff.clip(min=0),I_ff)
        g_i = - np.dot(self.W_ff.clip(max=0),I_ff)
        
        # Matching potential (equilibrium)
        V_m = (g_e*self.E_e+(g_i+self.g_inh)*self.E_i)/(g_e+g_i+self.g_inh)
        
        # Teacher-imposed firing rate
        r_m = util.act_fun(V_m,self.fun)
        
        return r_m, V_m