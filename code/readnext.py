#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:27:10 2021

@author: adkiran
"""
#this compares similiarity-based recomm vs cost_similarity based recomm.

import os, sys
import numpy as np
import scipy.stats as stats
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns


num_art_db = 500 #100  #num of articles in database
num_art_read = 5 #100  #num of articles read by user


###### value-distrib
mu_val = 0.5; std_val = 0.15
low_val = 0; high_val = 1
values = truncnorm(a=(low_val-mu_val)/std_val, b=(high_val-mu_val)/std_val, loc=mu_val, scale=std_val).rvs((num_art_db,1))
# values = np.random.uniform(low=low_val, high=high_val, size=(num_art_db, 1))


###### embedding distribution
ndim = 10 #dimension of embedding
mu_emb = 2.5; std_emb = 1
low_emb = 0; high_emb = 5
embeddings = np.random.uniform(low=low_emb, high=high_emb, size=(num_art_db, ndim))
# embeddings = truncnorm(a=(low_emb-mu_emb)/std_emb, b=(high_emb-mu_emb)/std_emb, loc=mu_emb, scale=std_emb).rvs((num_art_db, ndim))


###### Value matrices
U1 = np.zeros(shape=(num_art_db, num_art_db)) # purely 1/distance
U2 = np.zeros(shape=(num_art_db, num_art_db)) # val/distance

for row in range(num_art_db):
    for col in range(0, row+1):
        if row==col: continue
        dist = np.linalg.norm( embeddings[row,:]-embeddings[col,:] ) #dist between curr. and recom. article
        val = values[col,0] #val of recomm article

        U1[row, col] = U1[col, row] =  1/dist
        U2[row, col] = U2[col, row] = val/dist


###### random experiments
num_expts = 700000 #no. of random experiments
lift = 0

for _ in range(num_expts):
    ###### pick starting article
    I = np.random.randint(0, num_art_db) #picked by user
    
    val_sim = 0; val_cs = 0
    recommended_arts_sim = [I]; recommended_arts_cs = [I]
    
    i_sim = I; i_cs = I
    for j in range(num_art_read-1):
        ## similarity based
        arts_sim_tmp = U1[i_sim,:]
        sortinds_sim = np.argsort(-arts_sim_tmp) #desc
        sortinds_sim = [item for item in sortinds_sim if(item not in recommended_arts_sim) ]
    
        i_sim = sortinds_sim[0] #first
        recommended_arts_sim.append(i_sim)
        val_sim+= values[i_sim, 0]
    
        ## cost-similarity based
        arts_cs_tmp = U2[i_cs,:]
        sortinds_cs = np.argsort(-arts_cs_tmp) #desc
        sortinds_cs = [item for item in sortinds_cs if(item not in recommended_arts_cs) ]
    
        i_cs = sortinds_cs[0] #first
        recommended_arts_cs.append(i_cs)
        val_cs+= values[i_cs, 0]
    
    # print('recommended articles by sim:', recommended_arts_sim)
    # print('recommended articles by cost-sim:', recommended_arts_cs)
    print('\n')
    
    value_lift = 100*(val_cs-val_sim)/val_sim
    lift+= value_lift
    print('Value gained by sim: %2.3f'%(val_sim))
    print('Value gained by cost-sim: %2.3f'%(val_cs))
    print('value lift: %2.2f %%'%(value_lift))

print('\n\nFinal mean_value_lift after %d experiments: %2.2f %%'% (num_expts, lift/num_expts))
# Final mean_value_lift after 30000 experiments: 37.46 %
# Final mean_value_lift after 100000 experiments: 35.11 %
# Final mean_value_lift after 100000 experiments: 24.39 %
# Final mean_value_lift after 200000 experiments: 26.31 %
# Final mean_value_lift after 200000 experiments: 22.11 %
# Final mean_value_lift after 200000 experiments: 27.16 %
# Final mean_value_lift after 700000 experiments: 28.13 %