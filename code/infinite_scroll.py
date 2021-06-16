# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import spacy_universal_sentence_encoder

# COMMAND ----------

### Some useful functions
def cosine_dist(v1, v2):
  return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# COMMAND ----------

nlp = spacy_universal_sentence_encoder.load_model('en_use_md')

# COMMAND ----------

 

# COMMAND ----------

# DBTITLE 1,Main stuff starts from here:
#loads all the 70K articles
content_path = '/dbfs/FileStore/ewashington/hl-text-data/content_df.csv'
df = pd.read_csv(content_path)

# COMMAND ----------

#cleaning:

cleandf = df[ (df['summary'].notna()) &
             (df['post_title'].notna()) &
             (df['meta_title'].notna()) &
             (df['subvertical'].notna()) &
             (df['revenue']!=0) &
             (df['impressions']!=0)].copy()
akdf=cleandf.loc[0:70000,:].copy()
#del df, cleandf

# COMMAND ----------

#akdf.shape[0]
#df[['subvertical']].groupby(by=['subvertical']).size().reset_index(name='mycount')
# df.shape[0]- df[ (df['subvertical'].notna())].shape[0]

# COMMAND ----------

akdf['composite']=akdf['post_title']+' '+akdf['meta_title']+' '+akdf['summary']
akdf['tokenized'] = akdf['composite'].apply(nlp)
akdf['embeddings'] = akdf['tokenized'].apply(  lambda my_tokens: my_tokens.vector  )
akdf['values'] = akdf['revenue']/akdf['impressions']

# COMMAND ----------

## distribution of values in the overall articles

plt.figure(figsize=(7,4))
akdf['values'].hist(bins=50, density=False)
plt.xlabel(r'value (\$/imprs) $\longrightarrow$', fontsize=12);
plt.title('Distribution of article values')

# COMMAND ----------



# COMMAND ----------

#Setting parameters:

num_art_db = akdf.shape[0] #100  #num of articles in database
num_art_read = 5 #100  #num of articles read by user
r1 = 0.8 #0.6 #parameter for exponenent on cost
r2 = 15 # 100 #parameter for exponenent on d
disc_fact = 0.09 #0.9

# COMMAND ----------

 

# COMMAND ----------

###### random experiments for infinite_scroll
num_expts = 30000 #00 #00000 #no. of random experiments

val_sim = 0; val_cs = 0 #total value over all expts
mean_cos_dist_sim = []; mean_cos_dist_cs = [] #all cosine_sim over all expts
all_value_lifts = []

assigned_vals_sim = np.zeros((num_expts, num_art_read)); assigned_vals_cs = np.zeros((num_expts, num_art_read)) #vals of all assigned articles
assigned_cossim_sim = np.zeros((num_expts, num_art_read)); assigned_cossim_cs = np.zeros((num_expts, num_art_read)) #cossim of all assigned articles

for iterate in range(num_expts):
    ###### pick starting article
    
    if iterate%100==0: print('Status: %d/%d'%(iterate, num_expts) )
    I = np.random.choice(akdf.index.values) #picked by user
    print('-------------------------------------------------------------------------------------')
    print('Rand-expt #%d/%d'%(iterate, num_expts))
    print('\033[1m' + 'Current article being read:- \n \033[0m', akdf.loc[I:I].post_title.values)
    recommended_arts_sim = [I]; recommended_arts_cs = [I]

    tempdf = akdf[ akdf.subvertical ==  akdf.loc[I].subvertical ].copy() #copies the indices as well
    tempdf.drop(index=[I], inplace=True) #removing this current article

    curr_art = akdf.loc[I].embeddings
    tempdf['distances'] = tempdf['embeddings'].apply(lambda other_art: np.linalg.norm(curr_art-other_art))
    tempdf['cosine_sim'] = tempdf['embeddings'].apply(lambda other_art: cosine_dist(curr_art, other_art))
    tempdf['utility_1'] = 1/tempdf['distances']
    tempdf['utility_2'] = np.power(tempdf['values'], r1)/np.power(tempdf['distances'], r2)

    ## by distances -----------------------------------------------------
    tempdf.sort_values(by='utility_1', ascending=False, inplace=True)

    tmp_vals_sim = [tempdf.iloc[0:num_art_read]['values'].values]*np.array([np.power(disc_fact, i) for i in range(min(num_art_read, tempdf.shape[0] )) ])
    assigned_vals_sim[iterate, 0: len(tmp_vals_sim.ravel())] = tmp_vals_sim
    val_sim_temp = np.sum(tmp_vals_sim) #wghtd value
    val_sim+= val_sim_temp

    tmp_cossim_sim = tempdf.iloc[0:num_art_read]['cosine_sim'].values
    assigned_cossim_sim[iterate, 0: len(tmp_cossim_sim.ravel())] = tmp_cossim_sim
    mean_cos_dist_sim.append(tmp_cossim_sim.mean())

    recommended_arts_sim+= list(tempdf.iloc[0:num_art_read].index.values)

    print('\033[1m' + '\nRecommended using distance-based:-\033[0m')
    for i, item in enumerate(tempdf.iloc[0:num_art_read].post_title.values, 0): print(str(i+1)+'. '+'[disc_val=%2.5f, cossim=%2.4f]'%(tmp_vals_sim.ravel()[i], tmp_cossim_sim.ravel()[i])+ ' '+item)

    ## by value_distances ------------------------------------------------
    tempdf.sort_values(by='utility_2', ascending=False, inplace=True)

    tmp_vals_cs = [tempdf.iloc[0:num_art_read]['values'].values]*np.array([np.power(disc_fact, i) for i in range(min(num_art_read, tempdf.shape[0] )) ])
    assigned_vals_cs[iterate, 0: len(tmp_vals_cs.ravel())] = tmp_vals_cs
    val_cs_temp = np.sum(tmp_vals_cs) #wghtd value
    val_cs+= val_cs_temp

    tmp_cossim_cs = tempdf.iloc[0:num_art_read]['cosine_sim'].values
    assigned_cossim_cs[iterate, 0: len(tmp_cossim_cs.ravel())] = tmp_cossim_cs
    mean_cos_dist_cs.append(tmp_cossim_cs.mean())

    recommended_arts_cs+= list(tempdf.iloc[0:num_art_read].index.values)

    print('\033[1m' + '\nRecommended using value-based:-\033[0m')
    for i, item in enumerate(tempdf.iloc[0:num_art_read].post_title.values, 0): print(str(i+1)+'. '+'[disc_val=%2.5f, cossim=%2.4f]'%(tmp_vals_cs.ravel()[i], tmp_cossim_cs.ravel()[i])+ ' '+item)          

    tmp_val_lift = 100*(val_cs_temp-val_sim_temp)/val_sim_temp
    all_value_lifts.append(tmp_val_lift)
    print('\n Value: Old= %2.5f \t Value-new= %2.5f \t lift= %2.2f %% '%(val_sim_temp, val_cs_temp, tmp_val_lift))
    print('Mean cosine_similarity: Old= %2.3f \t new= %2.3f'%( mean_cos_dist_sim[-1], mean_cos_dist_cs[-1]))
print('-------------------------------------------------------------------------------------')


# COMMAND ----------

 

# COMMAND ----------



# COMMAND ----------

## Final result

value_lift = 100*(val_cs-val_sim)/val_sim
print('\033[1mAfter %d random-simulations: \033[0m'%(num_expts))
print('Value lift over purely-distance based recommendation: %2.2f %% '%(value_lift))
print('Mean cosine_similarity: Purely-dist=%2.3f, in value-dist:=%2.3f'%(np.mean(mean_cos_dist_sim), np.mean(mean_cos_dist_cs)))

# COMMAND ----------

 

# COMMAND ----------

## Distrib. of values in the assigned articles
plt.figure(figsize=(8,6))

wghtd = False
# wghtd = True

a1 = None
a2 = None

if wghtd:
  adjusted_vals_sim = assigned_vals_sim
  adjusted_vals_cs = assigned_vals_cs
  plt.ylabel(r'Discounted value $\rightarrow$', fontsize=16)
  plt.title('Mean of discounted-values in the recommended %d articles'%(num_art_read))
else:
  adjusted_vals_sim = assigned_vals_sim*np.array([np.power(disc_fact, -i) for i in range(num_art_read) ])
  adjusted_vals_cs = assigned_vals_cs*np.array([np.power(disc_fact, -i) for i in range(num_art_read) ])
  plt.ylabel(r'value $\rightarrow$', fontsize=16)
  plt.title('Mean of unweighted-values in the recommended %d articles'%(num_art_read))


plt.bar( np.arange(1, num_art_read+1), adjusted_vals_sim[a1:a2, :].mean(axis=0), color='b', width=0.25, label='distance based')
plt.bar( np.arange(1, num_art_read+1)+0.25, adjusted_vals_cs[a1:a2, :].mean(axis=0), color='r', width=0.25, label='value-based')

plt.xlabel(r'$N^{th}$ article $\longrightarrow$', fontsize=16)

plt.legend()
plt.show()


# COMMAND ----------

## Distrib. of cossine-similarities in the assigned articles

plt.figure(figsize=(8,6))
b1 = None
b2 = None


plt.bar( np.arange(1, num_art_read+1), assigned_cossim_sim[b1:b2, :].mean(axis=0), color='b', width=0.25, label='distance based')
plt.bar( np.arange(1, num_art_read+1)+0.25, assigned_cossim_cs[b1:b2, :].mean(axis=0), color='r', width=0.25, label='value-based')

plt.xlabel(r'$N^{th}$ article $\longrightarrow$', fontsize=16)
plt.ylabel(r'cossine-similarity $\rightarrow$', fontsize=16)
plt.legend()
plt.title('Mean cossine-similarities in the assigned %d articles'%(num_art_read))
plt.show()

# COMMAND ----------

# plt.figure(figsize=(7,4))
# for i in range(adjusted_vals_cs.shape[0]):
#   vec=adjusted_vals_cs[i,:]
#   #plt.plot()
#   plt.plot(vec, '.-')

# COMMAND ----------

 

# COMMAND ----------

## Distribution of simulations with non-zero lifts

lowlimitlift = 20
non_zero_lifts = [lift for lift in all_value_lifts if lift>lowlimitlift]
plt.figure(figsize=(47,6))
plt.hist(non_zero_lifts, bins=100, density=False)
plt.xlabel('lift observed in each random sim', fontsize=18)
plt.grid(); plt.show()

# COMMAND ----------

print(' %d/%d=%2.2f%% of the times we achieve a lift greater than %2.2f%%'%(len(non_zero_lifts), len(all_value_lifts), 100*len(non_zero_lifts)/len(all_value_lifts), lowlimitlift))

# COMMAND ----------

 

# COMMAND ----------

 

# COMMAND ----------

# akdf[['subvertical']].groupby(by=['subvertical']).size()

# COMMAND ----------

# tot = 0
# for subv in akdf.subvertical.unique():
#   tot+= (akdf[akdf.subvertical==subv].shape[0])**2 #3/akdf.shape[0]
# print('Tot:', tot*1e-6)

# COMMAND ----------

# akdf[akdf.post_title.str.match(pat='Can Aloe Vera Be')]

# COMMAND ----------



# COMMAND ----------


