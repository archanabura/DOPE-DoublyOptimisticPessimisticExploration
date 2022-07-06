#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 21:59:38 2022

@author: Anonymous
"""


#Imports
import numpy as np
import pandas as pd
from UtilityMethods_factored import utils
import matplotlib.pyplot as plt
import time
import os
import math
import pickle
import sys
import random


start_time = time.time()


#temp = sys.argv[1:]
#RUN_NUMBER = int(temp[0])

RUN_NUMBER = 10

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

#RUN_NUMBER = 0

#Initialize:
f = open('model_factored.pckl', 'rb')
[NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, DELTA] = pickle.load(f)
f.close()


f = open('solution-factored.pckl', 'rb')
[opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon] = pickle.load(f)
f.close()


f = open('base-factored.pckl', 'rb')
[pi_b, val_b, cost_b, q_b] = pickle.load(f)
f.close()

f = open('model_factored_abs.pckl', 'rb')
[P_abs, R_abs, C_abs, N_STATES_abs] = pickle.load(f)
f.close()

EPS = 1

M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2


Cb = cost_b[0, 0]

print(CONSTRAINT - Cb)


N_ACTIONS = 2

K0 = N_STATES**2*N_ACTIONS*EPISODE_LENGTH**3/((CONSTRAINT - Cb)**2)

print("printing K_0")
print(K0)



NUMBER_EPISODES = int(NUMBER_EPISODES)
NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)

STATES = np.arange(N_STATES)

ObjRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
ConRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))

NUMBER_INFEASIBILITIES = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))


L = math.log(6 * N_STATES**2 * EPISODE_LENGTH * NUMBER_EPISODES / DELTA)#math.log(2 * N_STATES * EPISODE_LENGTH * NUMBER_EPISODES * N_STATES**2 / DELTA)

for sim in range(NUMBER_SIMULATIONS):
    util_methods = utils(EPS, DELTA, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,CONSTRAINT,CONSTRAINT/5,P_abs,R_abs,C_abs,N_STATES_abs)
    ep_count = np.zeros((N_STATES, N_STATES))
    ep_count_p = np.zeros((N_STATES, N_STATES, N_STATES))
    ep_emp_reward = {}
    ep_emp_cost = {}
    
    for s in range(N_STATES):
        ep_emp_reward[s] = {}
        ep_emp_cost[s] = {}
        for a in range(N_ACTIONS):
            ep_emp_reward[s][a] = 0
            ep_emp_cost[s][a] = 0
    objs = []
    cons = []
    
    for episode in range(NUMBER_EPISODES):
        
        if episode <= K0:
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0)
            util_methods.update_empirical_rewards(ep_emp_reward, ep_emp_cost)
            util_methods.compute_confidence_intervals(L, 1)
            
        else:
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0)
            util_methods.update_empirical_rewards(ep_emp_reward, ep_emp_cost)
            util_methods.compute_confidence_intervals(L, 1)
            pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP(0, Cb)
            if log != 'Optimal':  
                pi_k = pi_b
                val_k = val_b
                cost_k = cost_b
                q_k = q_b
                #print(log)
            #else:
                #print(log)


        
        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[0, 0] - opt_value_LP_con[0, 0])
            ConRegret2[sim, episode] = max(0, cost_k[0, 0] - CONSTRAINT)
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = 1
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[0, 0] - opt_value_LP_con[0, 0])
            ConRegret2[sim, episode] = ConRegret2[sim, episode - 1] + max(0, cost_k[0, 0] - CONSTRAINT)
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1
        
        ep_count = np.zeros((N_STATES, N_STATES))
        ep_count_p = np.zeros((N_STATES, N_STATES, N_STATES))

        s = 0
        for h in range(EPISODE_LENGTH):
            prob = pi_k[s, h, :]
           
            a = int(np.random.choice(STATES, 1, replace = True, p = prob))
            next_state, rew, cost = util_methods.step(s, a, h)
            ep_count[s, a] += 1
            ep_count_p[s, a, next_state] += 1
            ep_emp_reward[s][a] += rew
            ep_emp_cost[s][a] += cost
            s = next_state
        if episode != 0 and episode%50000== 0:

            filename = 'opsrl-simple' + str(RUN_NUMBER) + '.pckl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []
        elif episode == NUMBER_EPISODES-1:
            filename = 'opsrl-simple' + str(RUN_NUMBER) + '.pckl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        

ObjRegret_mean = np.mean(ObjRegret2, axis = 0)
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)





title = 'DOPE' + str(RUN_NUMBER)
plt.figure()
plt.plot(range(NUMBER_EPISODES), ObjRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret')
plt.title(title)
plt.show()


plt.figure()
plt.plot(range(NUMBER_EPISODES), ConRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ConRegret_mean - ConRegret_std, ConRegret_mean + ConRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Constraint Regret')
plt.title(title)
plt.show()

