#!/usr/bin/env python
# coding: utf-8

import pandas as pd
actors = pd.read_csv('/Users/apillai2/Downloads/name.basics.tsv', sep='\t', header=0)

# # Apply Filters

actors = actors[(actors.birthYear > '1960') & (actors.primaryProfession.str.contains('act'))]
actors['titleCount'] = actors['knownForTitles'].str.count(',')+1
actors = actors[(actors.titleCount > 3)]
print(len(actors))
actors = actors[0:100]
actors = actors.reset_index(drop=True)
print(len(actors))

# # Get Unique Titles for Actors

# In[7]:


allTitles = ['tt0118571']
for i in actors['knownForTitles'].tolist():
    allTitles=allTitles+i.split(',')

uniqTitles = list(set(allTitles))



# # Scrape Related Fuctions

# In[8]:


from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
import re 
import urllib
import time


# In[9]:


#create a webdriver object and set options for headless browsing
options = Options()
options.headless = True
driver = webdriver.Chrome('./chromedriver',options=options)
#driver = webdriver.Chrome("/usr/local/bin/chromedriver",options=options)


# In[10]:


#uses webdriver object to execute javascript code and get dynamically loaded webcontent
def get_js_soup(url,driver):
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser') #beautiful soup object to be used for parsing html content
    return soup

#tidies extracted text 
def process_bio(bio):
    bio = bio.encode('ascii',errors='ignore').decode('utf-8')       #removes non-ascii characters
    bio = re.sub('\s+',' ',bio)       #repalces repeated whitespace characters with single space
    return bio

''' More tidying
Sometimes the text extracted HTML webpage may contain javascript code and some style elements. 
This function removes script and style tags from HTML so that extracted text does not contain them.
'''
def remove_script(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    return soup


#Checks if bio_url is a valid faculty homepage
def is_valid_homepage(bio_url,dir_url):
    if bio_url.endswith('.pdf'): #we're not parsing pdfs
        return False
    try:
        #sometimes the homepage url points to the same page as the faculty profile page
        #which should be treated differently from an actual homepage
        ret_url = urllib.request.urlopen(bio_url).geturl() 
    except:
        return False       #unable to access bio_url
    urls = [re.sub('((https?://)|(www.))','',url) for url in [ret_url,dir_url]] #removes url scheme (https,http or www) 
    return not(urls[0]== urls[1])

def scrape_movie_page(fac_url,driver):
    soup = get_js_soup(fac_url,driver)
    homepage_found = False
    bio_url = fac_url
    #profile_sec = soup.find('section',class_='main-content')
    profile_sec = soup.find('section',class_='article listo')
    if profile_sec == None:
        bio = ''
        print("Skipping bio : ",bio_url)
    else:
        bio = process_bio(profile_sec.get_text(separator=' '))
    #print("bio is ",bio_url)
    return bio_url,bio


# In[11]:


def write_lst(lst,file_):
    lst = lst.replace(' Plot Showing all 0 items Jump to: ','')
    lst = lst.replace(' Plot Showing all 1 items Jump to: Summaries (1) ','')
    lst = lst.replace(' Plot Showing all 2 items Jump to: Summaries (2) ','')
    lst = lst.replace(' Plot Showing all 3 items Jump to: Summaries (3) ','')
    lst = lst.replace(' Plot Showing all ','').replace(' items Jump to: Summaries (','')
    lst = lst.replace(') Synopsis (','').replace(' Synopsis','').replace(' Summaries','')
    #lst = lst[6:]
    lst = lst.replace('Edit ','')
    #lst = lst.replace(' Summaries ','\n\nSummaries\n')
    #lst = lst.replace(' Synopsis ','\n\nSynopsis\n')

    with open(file_,'w') as f:
        f.write(lst)


# # Scrape Movie plot summary and store in "movieFile" dir 

# In[12]:


#Scrape homepages of all urls
tot_urls = len(uniqTitles)
for i,l in enumerate(uniqTitles):
    link="https://www.imdb.com/title/"+l+"/plotsummary"
    print ('-'*20,'Scraping movie url {}/{}'.format(i+1,tot_urls),'-'*20)
    print(link)
    bio_urls, bios = [],[]
    bio_url,bio = scrape_movie_page(link,driver)
    if bio.strip()!= '' and bio_url.strip()!='':  
        bio_file = 'movieFile/'+l+'.txt'
        write_lst(bio,bio_file)
driver.close()


# # Concatenate movies for an actor and store as a single document

# In[13]:


for i in range(len(actors)):
    fl_1 = actors['nconst'][i]
    #dataAct=actors['primaryName'][i]+':\n'
    dataAct=actors['nconst'][i]+':\n'
    for fl_2 in actors['knownForTitles'][i].split(','):
        fl_2 = 'movieFile/'+fl_2+'.txt'
        with open(fl_2, 'r') as file:
            data = file.read().replace('\n \n', '')
            dataAct = dataAct+'\n'+data
    act_file = 'actorFile/'+fl_1+'.txt'
    write_lst(dataAct,act_file) 


# In[14]:


actors.to_csv('actors.csv',header=True,index=False)


# In[ ]:




