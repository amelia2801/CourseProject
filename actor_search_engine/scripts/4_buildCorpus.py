#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import glob
from os.path import exists


# # Get All Documents

# In[37]:


# All files ending with .txt
fileList = glob.glob("../movieFile/*.txt")


# In[38]:


## 'movieFile/tt0443701.txt',
docList = []
movList = []
movText = []
for fl in fileList:
    with open(fl, 'r') as file:
        data = file.read().replace('\n', '')
        dataSamp = data[:500]
        f1 = dataSamp.find(')')
        f2 = dataSamp.find(')',f1+1)
        fdiff = f2 - f1
        if fdiff <= 10:
            dataSamp1 = dataSamp[:f1].strip().strip(' ')
            dataSamp2 = dataSamp[f2+1:].strip().strip(' ')
            dataSamp = dataSamp1 +") : " + dataSamp2
        
        movname = fl.replace('../movieFile/','').replace('.txt','')
        data = movname+': '+data
        
    fl1 = '../movieFileTags/'+movname+'.txt'
    if exists(fl1):
        with open(fl1, 'r') as file:
            dataTags = file.read().replace('\n', '')
            data = data + dataTags

    docList.append(data)
    movList.append(movname)
    movText.append(dataSamp)
df = pd.DataFrame(movList)
df.columns = ['tconst']
df['mov_text'] = movText
df.head()


# In[39]:


print(len(df))
df.to_csv('../files/corpusList.csv',header=True,index=False)


# In[40]:


with open('../corpus/corpus.dat', 'w') as f:
    for item in docList:
        f.write("%s\n" % item)

