# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:18:13 2022

@author: passi
"""

import hiive.mdptoolbox.example
from hiive.mdptoolbox import mdp, example
from gym.envs.toy_text.frozen_lake import generate_random_map

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def init():
    np.random.seed(0)
    P, R = example.forest(S = 500)
    # R.shape  (100, 4), P.shape  (4, 100, 100)
    return P, R

# gamma = 0.5
for gamma in [0.5, 0.9]:
    
    policy = pd.DataFrame()
    iters = pd.DataFrame(columns = ['iters'])
    times = pd.DataFrame(columns = ['time'])
    mean_discrepancy = []
    
    print('ValueIteration')
    epsilons = [0.01, 0.001]
    for epsilon_ in epsilons:
        P, R = init()
        vi = mdp.ValueIteration(P, R, gamma, epsilon = epsilon_)
        vi.run()
        policy['vi' + str(epsilon_)] = vi.policy
        iters.loc['vi' + str(epsilon_)] = vi.iter
        times.loc['vi' + str(epsilon_)] = vi.time
    
    
    print('PolicyIteration')
    P, R = init()
    pi = mdp.PolicyIteration(P, R, gamma)
    pi.run()
    policy['pi'] = vi.policy
    iters.loc['pi'] = vi.iter
    times.loc['pi'] = vi.time
    
    
    print('QLearning')
    n_iters = [10000, 100000]
    for n_iter_ in n_iters:
        P, R = init()
        ql = mdp.QLearning(P, R, gamma, n_iter = n_iter_)
        ql.run()
        policy['ql' + str(n_iter_)] = ql.policy
        times.loc['ql' + str(n_iter_)] = ql.time
        # ql_df['Q'+ str(n_iter_)] = ql.Q
        mean_discrepancy.append(ql.mean_discrepancy)
        plt.plot(ql.mean_discrepancy)
        plt.xlabel('number of every 100 iterations')
        plt.ylabel('Vector of learned value discrepancy mean')
        plt.title('Q learning with n_iter = ' + str(n_iter_))
        plt.show()

