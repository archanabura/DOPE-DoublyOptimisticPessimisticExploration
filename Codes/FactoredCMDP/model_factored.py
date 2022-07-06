# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:40:31 2021

@author: Anonymous
"""

import numpy as np
import pickle
from UtilityMethods_factored import utils


states = np.arange(0,3)
nA = 2
actions = {0:[0,1],1:[0,1],2:[0,1]}


N_STATES = len(states)
print("printing number of states"+str(N_STATES))


P = {}
R = {}
C = {}


for s in states:
   P[s] = {}
   R[s] = {}
   C[s] = {}

   
   for a in range(0,2):
    P[s][a] = {}
    R[s][a] = 0
    C[s][a] = 0
    for s_1 in states:
        P[s][a][s_1] = 0
    if a == 0:
       P[s][a][s] = 1
    else:
        R[s][a] = -(s + 1)
        C[s][a] = 1
        if s < N_STATES-1:
            P[s][a][s+1] = 1
        else:
            P[s][a][0] = 1


for s in states:
    for a in range(0,2):
        R[s][a] = R[s][a]/3

#print(P[0][1][2])
print(R)
print(C)
print("printing matrix")
print(P)


#Creating abstract CMDP

states_abs = np.arange(0,1)
nA_abs = 2


N_STATES_abs = len(states_abs)
print("printing number of states"+str(N_STATES_abs))


P_abs = {}
R_abs= {}
C_abs = {}


P_abs[0] = {}
P_abs[0][0] = {}
P_abs[0][1] = {}
P_abs[0][0][0] = 1
P_abs[0][1][0] = 1

R_abs[0] = {}
R_abs[0][0] = 0
R_abs[0][1] = 2

C_abs[0] = {}
C_abs[0][0] = 0
C_abs[0][1] = 1
     

#print(P[0][1][2])
print(R_abs)
print(C_abs)
print(P_abs)



EPISODE_LENGTH = 6

CONSTRAINT = EPISODE_LENGTH/2

C_b = CONSTRAINT/5   #Change this if you want different baseline policy.

NUMBER_EPISODES = 1e5
NUMBER_SIMULATIONS = 1


EPS = 0.01
M = 0
delta = 0.01

util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,CONSTRAINT,C_b, P_abs, R_abs, C_abs, N_STATES_abs)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0)
print("return values")
print(opt_cost_LP_con[0,0])
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0)
f = open('solution-factored.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()


util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,C_b,C_b,P_abs, R_abs, C_abs, N_STATES_abs)
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
f = open('base-factored.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()




f = open('model_factored.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, delta], f)
f.close()

f = open('model_factored_abs.pckl', 'wb')
pickle.dump([P_abs, R_abs, C_abs, N_STATES_abs], f)
f.close()

print('*******')
print(opt_value_LP_uncon[0, 0],opt_cost_LP_uncon[0,0])
print(opt_value_LP_con[0, 0],opt_cost_LP_con[0,0])
print(value_b[0, 0],cost_b[0,0])
