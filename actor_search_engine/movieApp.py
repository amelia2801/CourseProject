#!/usr/bin/env python
# coding: utf-8
from requests import get
from requests.exceptions import RequestException
import numpy as np
import pandas as pd
import glob
import re
import math
import sys
import time
import metapy
import pytoml
import gc


import os
import requests
import operator
from flask import Flask, render_template, request
from collections import Counter
from datetime import datetime

def preprocess_corpus():
    cfg = 'config.toml'
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    return idx

def getActor(df):
    corpusList=pd.read_csv('files/corpusList.csv')
    df = df.merge(corpusList,how='inner')
    df_actor=pd.read_csv('files/actors.csv')
    df_actor =  df_actor.merge(df,how='inner')
    df_actor = df_actor[df_actor.doc_scores > 1]
    df_actor = df_actor.sort_values('doc_scores',ascending=False)

    df_actor['cum_rank'] = df_actor.groupby('nconst')['doc_scores'].rank("dense", ascending=False)
    df_actor = df_actor[df_actor.cum_rank <= 4]
    df_actor['cum_score'] = df_actor.groupby('nconst')['doc_scores'].transform("sum")
    df_actor = df_actor.sort_values(['cum_score','doc_scores'],ascending=False)
    df_actor = df_actor[:50].reset_index()

    df_actor['movLink'] = 'https://www.imdb.com/title/'+df_actor['tconst']+'/plotsummary'
    df_actor['actorLink'] = 'https://www.imdb.com/name/'+df_actor['nconst']
    df_actor = df_actor[['nconst','primaryName','mov_text','actorLink','movLink','cum_score']]

    df_actor_mov = df_actor.groupby(['nconst'])['movLink'].apply(list).reset_index()
    df_actor = df_actor.groupby(['nconst','primaryName','cum_score','actorLink'])['mov_text'].apply(list).reset_index()

    if len(df_actor) > 0:
        df_actor = df_actor.merge(df_actor_mov,how='inner')
        df_actor = df_actor.sort_values('cum_score',ascending=False).reset_index(drop=True)
        df_actor = df_actor[['primaryName','mov_text','actorLink','movLink']][:12]
    else:
        df_actor = pd.DataFrame([['No Results Found','','https://www.imdb.com/','']])

    del corpusList
    gc.collect()

    return df_actor


def getval(inp):

    print('Running queries',inp)
    logTxt = datetime.now().strftime("%Y%m%d %H:%M:%S")+" : "+inp+"\n"
    with open("logs/queries.txt", "a") as file_object:
        file_object.write(logTxt)

    query = metapy.index.Document()
    query.content(inp.strip())

    ranker = metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=500)

    idx = preprocess_corpus()
    results = ranker.score(idx, query, 100)
    del idx
    gc.collect()

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
    df_actor = getActor(df)

    df_actor['query']=inp

    out=df_actor.values.tolist()

    del df_actor,df,docList_df
    gc.collect()
    
    return out
    

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        inp = request.form['inp']
        results=getval(inp)

    return render_template('index.html', errors=errors, results=results)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, threaded=False)
