"""
Analyze trained network and produce figures.
"""

import os
from pathlib import Path
import pickle
import utilities as util
import matplotlib.pyplot as plt
import numpy as np

# Determine parameters to load the appropriate network
params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 1,       # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_in': 1,           # size of patterns
    'eta': 1,         # learning rate
    'n_trial': 1e2,      # number of trials
    'train': True,       # whether to train network or not
    'W_rec': None,       # recurrent weights of associative network
    'W_ff': None,        # feedforward weights to associative neurons
    'W_fb': None,        # feedback weights to associative neurons
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'est_every': True,  # whether to estimate US and reward after every trial
    'rule': 'Pred',      # learning rule used in associative network
    'run': 0,            # number of run for many runs of same simulation
    'v': 2,              # agent's constant velocity, in m/s
    'x_dim': 10,         # x dimension of track, in m
    'y_dim': None,       # y dimension of track, in m
    'sigma': 1,          # spatial extent of place cell receptive field
    'f_max': .1,         # maximum firing rate, in kHz
    'neu_den': 1,        # lattice density of place cells, in neurons/m
    'tau_lp': 50,        # time constant of additional PSP filtering, in ms
    'tau_eff': 1500      # effective bootstrapping time constant, in ms
    }


# Load network
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
filename = util.filename(params) + 'LinTrack'


with open(data_path+filename+'.pkl', 'rb') as f:
    net = pickle.load(f)

# Fontsize appropriate for plots
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title


# Obtain results
r = net.r
t = np.linspace(0,params['x_dim']/params['v'],np.size(r,2))

plt.plot(t,1000*r[0,:,0])
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (spikes/s)')
plt.title('Trial 1')
plt.show()

plt.plot(t,1000*r[99,:,0])
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (spikes/s)')
plt.title('Trial 100')
plt.show()

#plt.savefig('Cond.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('Cond.eps',bbox_inches='tight',format='eps',dpi=300)