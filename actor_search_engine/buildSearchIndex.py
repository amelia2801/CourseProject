#!/usr/bin/env python
# coding: utf-8

# In[9]:


import math
import sys
import time

import metapy
import pytoml

import pandas as pd


def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """

    return metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=500)


if __name__ == '__main__':


    cfg = 'config.toml'
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)

    query = metapy.index.Document()
    print('Running queries')
    line = 'crazy rich asians'
    query.content(line.strip())
    results = ranker.score(idx, query, 500)


# In[10]:


docList_tmp = []
with open("corpus/corpus.dat", "r") as f:
    for line in f:
        docList_tmp.append(line.split(':')[0])
docList_df = pd.DataFrame(docList_tmp)
docList_df.columns = ['tconst']
docList_df.to_csv('files/docList_df.csv',header=True,index=False)
docList_df.head()


# In[11]:


docList_df=pd.read_csv('files/docList_df.csv')
docList = docList_df['tconst'].to_list()

score_out = []
for num,(d_id, score) in enumerate(results):
    t_id = docList[d_id]
    score_out.append([t_id,score])
    
if len(score_out) == 0:
    score_out = [['t0',0]]
    
df = pd.DataFrame(score_out)
df.columns=['tconst','doc_scores']

corpusList=pd.read_csv('files/corpusList.csv') #.set_index('nconst')
df = df.merge(corpusList,how='inner')
df_actor=pd.read_csv('files/actors.csv')
df_actor =  df_actor.merge(df,how='inner')
df_actor = df_actor[df_actor.doc_scores > 0.1]
df_actor = df_actor.sort_values('doc_scores',ascending=False)

df_actor['cum_rank'] = df_actor.groupby('nconst')['doc_scores'].rank("dense", ascending=False)
df_actor = df_actor[df_actor.cum_rank <= 4]
df_actor['cum_score'] = df_actor.groupby('nconst')['doc_scores'].transform("sum")
df_actor = df_actor.sort_values(['cum_score','doc_scores'],ascending=False)
df_actor = df_actor[:20].reset_index()

df_actor['movLink'] = 'https://www.imdb.com/title/'+df_actor['tconst']+'/plotsummary'
df_actor['actorLink'] = 'https://www.imdb.com/name/'+df_actor['nconst']
df_actor = df_actor[['nconst','primaryName','mov_text','actorLink','movLink','cum_score']]
df_actor_mov = df_actor.groupby(['nconst'])['movLink'].apply(list).reset_index()
df_actor = df_actor.groupby(['nconst','primaryName','cum_score','actorLink'])['mov_text'].apply(list).reset_index()


# In[12]:


for i in range(len(df_actor)):
    actor = df_actor.iloc[i]['primaryName']
    movies = df_actor.iloc[i]['mov_text']
    print("**",actor)
    for m in movies:
        print(m)
        print("")
    print("=================================================")

