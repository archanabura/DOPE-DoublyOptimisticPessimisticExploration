#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:40:31 2021

@author: Anonymous
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UtilityMethods_Queue import utils
import sys
import pickle
import time
import pulp as p
import math
from copy import copy

c_0 = 1
c_1 = 1

mu_1 = 0.9
mu_2 = 0.1

BUFFER_SIZE = 1.0
lambda_m = 0.8

def P_1(a):
    if a == 0:
        return mu_1*lambda_m + (1-mu_1) *(1-lambda_m)
    else:
        return mu_2*lambda_m + (1-mu_2) * (1-lambda_m)

def P_2(a):
    if a == 0:
        return mu_1*(1-lambda_m)
    else:
        return mu_2*(1-lambda_m)

def P_3(a):
    if a == 0:
        return (1-mu_1)*lambda_m 
    else:
        return (1-mu_2)*lambda_m
    
def objcost(x):
    if x == 0:
        return c_0
    return 0

def concost(a):
    if a == 0:
        return c_1
    return 0


delta = 0.01

buffer_values = np.arange(0,BUFFER_SIZE,0.1)
nA = 2


N_STATES = len(buffer_values)
print("printing number of states"+str(N_STATES))

term_reward = np.zeros(N_STATES)
P = {}
R = {}
C = {}
actions = {}
for s in buffer_values:
   l = int(np.round(s/0.1))
   P[l] = {}
   R[l] = {}
   C[l] = {}
   actions[l] = [0,1]
   
   for a in range(0,2):
    P[l][a] = {}
    for s_1 in buffer_values:
        m = int(np.round(s_1/0.1))
        P[l][a][m] = 0
    R[l][a] = objcost(s)
    C[l][a] = concost(a)

    next_s = s
    P[l][a][l] += P_1(a)
    
    
    next_s = s+0.1
    if(next_s > BUFFER_SIZE-0.1):
        next_s = BUFFER_SIZE - 0.1
    m = int(np.round(next_s/0.1))
    P[l][a][m] += P_2(a)
    
    next_s = s-0.1
    if(next_s < 0):
        next_s = 0
    m = int(np.round(next_s/0.1))
    P[l][a][m] += P_3(a)

#print(P[0][1][2])
print(R)
print(C)
print(P[0][1])



EPISODE_LENGTH = 10

CONSTRAINT = EPISODE_LENGTH/2

C_b = CONSTRAINT/5   #Change this if you want a different baseline policy.

NUMBER_EPISODES = 1e5
NUMBER_SIMULATIONS = 1


EPS = 0.01
M = 0

util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,CONSTRAINT,C_b)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0)
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0)
f = open('solution-queue.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()


util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,C_b,C_b)
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
f = open('base-queue.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()




f = open('model-queue.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, delta], f)
f.close()

print('*******')
print(opt_value_LP_uncon[0, 0],opt_cost_LP_uncon[0,0])
print(opt_value_LP_con[0, 0],opt_cost_LP_con[0,0])
print(value_b[0, 0],cost_b[0,0])
