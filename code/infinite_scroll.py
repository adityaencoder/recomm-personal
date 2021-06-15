# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
# import spacy_universal_sentence_encoder

### Some useful functions
def cosine_dist(v1, v2):
  return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# nlp = spacy_universal_sentence_encoder.load_model('en_use_md')

content_path = './data/akdf.csv'
df = pd.read_csv(content_path)

#cleaning:
cleandf = df[ (df['summary'].notna()) &
             (df['post_title'].notna()) &
             (df['meta_title'].notna()) &
             (df['subvertical'].notna()) &
             (df['revenue']!=0)].copy()

akdf = cleandf.copy()
del df, cleandf

akdf['composite']=akdf['post_title']+' '+akdf['meta_title']+' '+akdf['summary']
akdf['tokenized'] = akdf['composite'].apply(nlp)
akdf['embeddings'] = akdf['tokenized'].apply(  lambda my_tokens: my_tokens.vector  )
akdf['values'] = akdf['revenue']

###########################################
###########################################
## Copy from local notebook henceforth

num_art_db = akdf.shape[0] #100  #num of articles in database
num_art_read = 5 #100  #num of articles read by user
r1 = 0.5 #parameter for exponenent on cost
r2 = 100 # 0 #parameter for exponenent on d
disc_fact = 0.9


###### random experiments for infinite_scroll
num_expts = 100 #00000 #no. of random experiments
lift = 0

val_sim = 0; val_cs = 0
mean_cos_dist_sim = []; mean_cos_dist_cs = []
for iterate in range(num_expts):
    ###### pick starting article
    if iterate%100==0: print('Status: %d/%d'%(iterate, num_expts) )
    I = np.random.choice(akdf.index.values) #picked by user
    recommended_arts_sim = [I]; recommended_arts_cs = [I]

    tempdf = akdf[ akdf.subvertical ==  akdf.loc[I].subvertical ].copy() #copies the indices as well
    tempdf.drop(index=[I], inplace=True) #removing this current article
    
    curr_art = akdf.loc[I].embeddings
    tempdf['distances'] = tempdf['embeddings'].apply(lambda other_art: np.linalg.norm(curr_art-other_art))
    tempdf['cosine_sim'] = tempdf['embeddings'].apply(lambda other_art: cosine_dist(curr_art, other_art))
    tempdf['utility_1'] = 1/tempdf['distances']
    tempdf['utility_2'] = np.power(tempdf['values'], r1)/np.power(tempdf['distances'], r2)
    
    ## by distances ------------
    tempdf.sort_values(by='utility_1', ascending=False, inplace=True)
    val_sim_temp = tempdf.iloc[0:num_art_read]['values'].values.sum()
    val_sim+= val_sim_temp
    recommended_arts_sim+= list(tempdf.iloc[0:num_art_read].index.values)
    mean_cos_dist_sim.append(tempdf.iloc[0:num_art_read]['cosine_sim'].values.mean())
    
    print('\n Using distance: \n', akdf.loc[I:I].post_title.values)
    print(tempdf.iloc[0:num_art_read].post_title.values)

    ## by cost_distances ------------
    tempdf.sort_values(by='utility_2', ascending=False, inplace=True)
    val_cs_temp = tempdf.iloc[0:num_art_read]['values'].values.sum()
    val_cs+= val_cs_temp
    recommended_arts_cs+= list(tempdf.iloc[0:num_art_read].index.values)
    mean_cos_dist_cs.append(tempdf.iloc[0:num_art_read]['cosine_sim'].values.mean())

    print('\n Using value_distance: \n', akdf.loc[I:I].post_title.values)
    print(tempdf.iloc[0:num_art_read].post_title.values)

    print('\n Value-dist: %2.3f \t Value-cost-sim: %2.3f \t lift: %2.2f %% '%(val_sim_temp, val_cs_temp, 100*(val_cs_temp-val_sim_temp)/val_sim_temp))
    print('Mean cosine_similarity in dist=%2.3f, in value_dist:=%2.3f'%(np.mean(mean_cos_dist_sim[-1]), np.mean(mean_cos_dist_cs[-1])))
    print('-------------------------------------------------------------')

value_lift = 100*(val_cs-val_sim)/val_sim
lift+= value_lift
print('\n\nValue lift over purely-distance based recommendation: %2.2f %% '%(value_lift))
print('Mean cosine_similarity in dist=%2.3f, in value_dist:=%2.3f'%(np.mean(mean_cos_dist_sim), np.mean(mean_cos_dist_cs)))

