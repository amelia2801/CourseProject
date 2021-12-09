#!/usr/bin/env python
# coding: utf-8

# # Read actors File

# In[42]:


import pandas as pd
from os.path import exists
import glob
actors = pd.read_csv('../files/actors.csv', header=0)


# # Apply Filters

# In[43]:


print("Actor Count before Filter:",len(actors))
#actors = actors[0:1000]
actors = actors.reset_index(drop=True)
print(len(actors))
actors.head()


# # Check Samples

# # Get Unique Titles for Actors

# In[44]:


uniqTitles = list(set(actors['tconst']))
uniqTitles.sort()
print(len(uniqTitles))


# # Scrape Related Fuctions

# In[45]:


from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
import re 
import urllib
import time


# In[46]:


#create a webdriver object and set options for headless browsing
options = Options()
options.headless = True
driver = webdriver.Chrome('../chromedriver',options=options)


# In[47]:


#uses webdriver object to execute javascript code and get dynamically loaded webcontent
def get_js_soup(url,driver):
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser') #beautiful soup object to be used for parsing html content
    return soup

#tidies extracted text 
def process_mov(mov):
    mov = mov.encode('ascii',errors='ignore').decode('utf-8')       #removes non-ascii characters
    mov = re.sub('\s+',' ',mov)       #repalces repeated whitespace characters with single space
    return mov

''' More tidying
Sometimes the text extracted HTML webpage may contain javascript code and some style elements. 
This function removes script and style tags from HTML so that extracted text does not contain them.
'''
def remove_script(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    return soup


#Checks if bio_url is a valid faculty homepage
def is_valid_homepage(mov_url,dir_url):
    if mov_url.endswith('.pdf'): #we're not parsing pdfs
        return False
    try:
        #sometimes the homepage url points to the same page as the faculty profile page
        #which should be treated differently from an actual homepage
        ret_url = urllib.request.urlopen(mov_url).geturl() 
    except:
        return False       #unable to access bio_url
    urls = [re.sub('((https?://)|(www.))','',url) for url in [ret_url,dir_url]] #removes url scheme (https,http or www) 
    return not(urls[0]== urls[1])

def scrape_movie_page(mov_url,driver):
    soup = get_js_soup(mov_url,driver)
    homepage_found = False
    #profile_sec = soup.find('section',class_='main-content')
    profile_sec = soup.find('div',class_='article listo')
    if profile_sec == None:
        mov = ''
        print("Skipping mov : ",mov_url)
    else:
        mov = process_mov(profile_sec.get_text(separator=' '))
    return mov_url,mov


# In[48]:


def write_lst(lst,file_):
    lst = lst.replace(' Plot Keywords Sort By: Relevance Alphabetical Showing all ',' ')
    lst = lst.replace(' found this relevant Relevant? Yes No ',' ')
    lst = lst.replace(' Is this relevant? Relevant? Yes No ',' ')
    lst = lst.replace(' 1 of 1 ',' ').replace(' 0 of 1 ',' ')
    lst = lst.replace(' 2 of 2 ',' ').replace(' 0 of 2 ',' ').replace(' 1 of 2 ',' ')
    lst = lst.replace(' 3 of 3 ',' ').replace(' 0 of 3 ',' ').replace(' 1 of 3 ',' ').replace(' 2 of 3 ',' ')
    lst = lst.replace(' Plot Showing all ','').replace(' items Jump to: Summaries (','')
    lst = lst.replace('  ',' ').replace('   ',' ')

    lst = lst.replace('Edit ','')


    with open(file_,'w') as f:
        f.write(lst)


# # Scrape Movie Tags and store in "movieFileTags" dir 

# In[49]:


fileList = glob.glob("../movieFileTags/*.txt")

if len(fileList) <= 0:
    last_scraped = 'a'
else:
    last_scraped = max(fileList).replace('../movieFileTags/','').replace('.txt','')

print(last_scraped)
print(len(uniqTitles))
uniqTitles = sorted(i for i in uniqTitles if i > last_scraped)
print(len(uniqTitles),len(fileList))


# In[50]:


#Scrape homepages of all urls
tot_urls = len(uniqTitles)
for i,l in enumerate(uniqTitles):
    link="https://www.imdb.com/title/"+l+"/keywords"
    print ('-'*20,'Scraping movie url {}/{}'.format(i+1,tot_urls),'-'*20)
    print(link)
    mov_urls, movs = [],[]
    mov_url,mov = scrape_movie_page(link,driver)
    if len(mov) < 10:
        print("SKIPPING")
    else:
        if mov.strip()!= '' and mov_url.strip()!='':
            mov_file = '../movieFileTags/'+l+'.txt'
            write_lst(mov,mov_file)
driver.close()


# In[51]:


fileList = glob.glob("../movieFileTags/*.txt")
activeMovs = []
for i in fileList:
    activeMovs.append(i.replace('../movieFileTags/','').replace('.txt',''))
    
df_act_movs = pd.DataFrame(activeMovs)
df_act_movs.columns = ['tconst']
df_act_movs.head()


# In[52]:


print(len(actors))
actors = actors.merge(df_act_movs,how='inner')
print(len(actors))


# # Prepare actors list for scraped movies

# In[53]:


print(len(actors))
actors.to_csv('../files/actors.csv',header=True,index=False)
actors.head()


# In[54]:


print(len(df_act_movs))


# In[ ]:




