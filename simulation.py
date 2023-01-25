# -*- coding: utf-8 -*-
"""
code to show how the performance depends on the used model parameters
this code runs one simulation:
one random set of model parameters is chosen (between given bounds)
for a given experimental design (size grid, number of trials),
one round is simulated
and this 100 times to get less noisy values for 
the average accumulated reward per trial
"""

import random
from create_grids import bivariate
from solving_models import Learnsolver
from csv import DictWriter


"""
fill in parameters
"""

#grid parameters
W = 15 #width of grid
L = W  #length of grid
l = 2
max_r = 100
min_r = 0

nr_trials = 30

runs = 100

file_name = "performance{}x{}_{}l{}.csv".format(W,L,nr_trials,l)

bounds_beta = (0.0, 1.0)
bounds_tau = (0.01, 0.2)
bounds_lfit = (0.1, 40)

#generate a random value for each model parameter
beta = random.uniform(bounds_beta[0], bounds_beta[1])
tau = random.uniform(bounds_tau[0], bounds_tau[1])
l_fit = random.uniform(bounds_lfit[0], bounds_lfit[1])
    
total_performance = 0
for run in range(0, runs):
        
    """"
    step 1: generate the grid
    """    
    bandit = bivariate(W, L, l, max_r, min_r)
          
    """
    step 2: one round
    """    
    total_r = Learnsolver(bandit, W, L, nr_trials, max_r, min_r, l_fit, beta, tau) 
    performance = total_r/nr_trials
        
    total_performance += performance
    

average = total_performance/runs
results = {"l_fit": l_fit, "beta": beta, "tau": tau, "performance": average}

'''
write out the performance for this parameter set
'''
field_names=["l_fit", "beta", "tau", "performance"]
#open CSV file in append mode
with open(file_name, 'a') as f_object:
	dictwriter_object = DictWriter(f_object, fieldnames=field_names)
	dictwriter_object.writerow(results)
	f_object.close()
