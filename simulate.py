"""
Run simulations and store results.
"""

import os
from pathlib import Path
import main
import utilities as util
import pickle


# Parameters

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
    'tau_lp': 100,        # time constant of additional PSP filtering, in ms
    'tau_eff': 1500      # effective bootstrapping time constant, in ms
    }

# Save directory
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
filename = util.filename(params) + 'LinTrack'

# Run simulation
net = main.ActorCriticNN(params)

net.simulate()

# Save results
with open(data_path + filename + '.pkl','wb') as f:
    pickle.dump(net, f)