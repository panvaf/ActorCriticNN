"""
Associative network class.
"""

import numpy as np
import util

# Define dynamics of associative network

class assoc_net:
    
    def __init__(self,params):
        # Initialize network
        
        # Constants
        self.dt = params['dt']
        self.n_sigma = params['n_sigma']
        self.n_neu = int(params['n_assoc'])
        self.tau_s = params['tau_s']
        self.n_in = int(params['n_in'])
        self.n_fb = int(params['n_fb'])
        self.eta = params['eta']
        self.dale = params['dale']
        self.I_inh = params['I_inh']
        self.rule = params['rule']
        self.alpha = self.dt/self.tau_s
        
        # Weights
        self.W_rec = np.zeros((self.n_neu,self.n_neu))
        self.W_ff = np.ones((self.n_neu,self.n_in))
        self.W_fb = np.random.normal(0,np.sqrt(1/self.n_neu),(self.n_assoc,self.n_fb))
        
        # Initial conditions
        self.V_d = np.random.uniform(0,1,self.n_assoc) 
        self.V = np.random.uniform(0,1,self.n_assoc)
        self.I_d = np.zeros(self.n_assoc)
        self.Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_in))
        self.PSP = np.zeros(self.n_assoc+self.n_in)
        self.I_PSP = np.zeros(self.n_assoc+self.n_in)
        self.g_e = np.zeros(self.n_assoc)
        self.g_i = np.zeros(self.n_assoc)
        self.r = np.random.uniform(0,.15,self.n_assoc)
        
    
    
    def dynamics(self,I_ff,I_fb,W_rec,W_ff,W_fb,tau_l=20,gD=.2,gL=.1,E_e=14/3,E_i=-1/3):
        # Network dynamics
    
        # units in ms or ms^-1
        c = gD/gL
        
        # Create noise that will be added to all origins of input
        n = np.random.normal(0,self.n_sigma,self.n_neu)
        n_d = np.random.normal(0,self.n_sigma,self.n_neu)
        
        # input to the dendrites
        self.I_d += (- self.I_d + np.dot(W_rec,self.r) + np.dot(W_fb,I_fb) + self.I_inh + n_d)*self.alpha
        
        # Dentritic potential is a low-pass filtered version of the dentritic current
        self.V_d += (-self.V_d+self.I_d)*self.dt/tau_l
        
        # Time-dependent somatic conductances
        self.g_e += (-self.g_e + np.dot(W_ff.clip(min=0),I_ff))*self.alpha
        self.g_i += (-self.g_i - np.dot(W_ff.clip(max=0),I_ff))*self.alpha
        
        # Input to the soma (teacher signal)
        I = self.g_e*(E_e-self.V) + (self.g_i+self.g_sh)*(E_i-self.V)
        
        # Somatic voltage
        self.V += (-self.V + c*(self.V_d-self.V) + I/gL + n)*self.dt/tau_l
        
        # Firing rate
        r = util.act_fun(self.V,self.fun)
        
        # Dendritic prediction of somatic voltage
        V_ss = self.V_d*gD/(gD+gL)
        
        # Discrepancy between dendritic prediction and actual firing rate
        self.error = r - util.act_fun(V_ss,self.fun)
        
        # Compute PSP for every dendritic input to associative neurons
        r_in = np.concatenate((r,I_fb))
        self.I_PSP += (- self.I_PSP + r_in)*self.alpha
        self.PSP += (- self.PSP + self.I_PSP)*self.dt/tau_l



    def learn_rule(self,filt=False,tau_d=100):
        # Learning dynamics
        
        # Weight update
        PI = np.outer(self.error,self.PSP)
       
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


    def ss_fr(self,I_ff,g_inh,fun,E_e=14/3,E_i=-1/3):
        # Find instructed steady-state firing rate for given feedforward input
        
        # Steady-state somatic conductances
        g_e = np.dot(self.W_ff.clip(min=0),I_ff)
        g_i = - np.dot(self.W_ff.clip(max=0),I_ff)
        
        # Matching potential (equilibrium)
        V_m = (g_e*E_e+(g_i+g_inh)*E_i)/(g_e+g_i+g_inh)
        
        # Teacher-imposed firing rate
        r_m = util.act_fun(V_m,fun)
        
        return r_m, V_m