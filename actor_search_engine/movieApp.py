#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from rank_bm25 import *

######    

actors = pd.read_csv('actors.csv').set_index('nconst')
docList = []

def preprocess_corpus():
    import glob
    # All files ending with .txt
    fileList = glob.glob("actorFile/*.txt")

    for fl in fileList:
        with open(fl, 'r') as file:
            data = file.read().replace('\n', '')
        docList.append(data)

    tokenized_corpus = [doc.split(" ") for doc in docList]

    global bm25 
    bm25 = BM25Okapi(tokenized_corpus)
    

def getval(inp):
    query = inp ## Enter search query
    tokenized_query = query.split(" ")

    rankedDocs = bm25.get_top_n(tokenized_query, docList, n=5)

    actorOut = []
    actorLink = []
    movieOut = []
    movLinks = []
    for i in rankedDocs:
        nmLoc = i.find(': ')
        i_Val=i[0:nmLoc]
        i_Loc = actors.index.get_loc(i_Val)
        i_Loc_df = actors.iloc[i_Loc]
        actor = i_Loc_df['primaryName']
        actorOut.append(actor)
        actorLink.append('https://www.imdb.com/name/'+i_Val)
        movies = i_Loc_df['knownForTitles']
        movs = []
        movLink = []
        for m in movies.split(','):
            fl_2 = 'movieFile/'+m+'.txt'
            with open(fl_2, 'r') as file:
                data = file.read() #.replace('\n \n', '')
                data = data[1:800] + '\n' + ' ' + '\n'
            movs.append(data)
            movLink.append('https://www.imdb.com/title/'+m+'/plotsummary')
        movLinks.append(movLink)
        movieOut.append(movs)
    
    dfl=pd.DataFrame()
    dfl['actorOut']=actorOut
    dfl['movieOut']=movieOut
    dfl['actorLink']=actorLink
    dfl['movLink']=movLinks
    dfl['query']=query
    out=dfl.values.tolist()
    
    return out
    


# In[4]:


##getOCRAval('https://www.coursera.org/search?query=java&%5Bpage%5D=2&')


# In[ ]:

from flask import Flask, render_template, request

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
    preprocess_corpus()
    app.run(host="0.0.0.0", port=80)
