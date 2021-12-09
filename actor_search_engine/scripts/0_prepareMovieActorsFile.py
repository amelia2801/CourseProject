#!/usr/bin/env python
# coding: utf-8

# # Processing Logic
# - Down load IMDb Datasets - name.basics.tsv, title.principals.tsv, title.akas.tsv, title.basics.tsv and title.ratings.tsv.
# - The dataset files are accessed and downloaded from https://datasets.imdbws.com/. 
# - Apply filters and prepare output with actor and movie IDs
# 
# ## Filters
# 
# - Actors born after 1940 and actor atleast associated with 4 movies
# - Region should be US
# - Atleast 2000 user votes
# 

# ## Download and Process Actors Information
# 
# ***name.basics.tsv.gz – Contains the following information for names:***
# 
# - nconst (string) - alphanumeric unique identifier of the name/person
# - primaryName (string)– name by which the person is most often credited
# - birthYear – in YYYY format
# - deathYear – in YYYY format if applicable, else '\N'
# - primaryProfession (array of strings)– the top-3 professions of the person
# - knownForTitles (array of tconsts) – titles the person is known for
# 
# **Following filters applied**
# 
# - Actors born after 1940
# - Atlease 4 movies
# 

import pandas as pd
actors = pd.read_csv('../sourceFiles/name.basics.tsv', sep='\t', header=0)
print("Actor Count before Filter:",len(actors))
actors=actors[(actors.birthYear != '\\N')]
actors = actors[(actors.birthYear > '1940') & (actors.primaryProfession.str.contains('act'))]

actors['titleCount'] = actors['knownForTitles'].str.count(',')+1
print("Actor Count after Filter:",len(actors))
actors = actors[(actors.titleCount > 3)]
actors = actors[['nconst','primaryName']]
actors.head()


# ## Download and Process principal actors for a given movie
# 
# ***title.principals.tsv.gz – Contains the principal cast/crew for titles***
# - tconst (string) - alphanumeric unique identifier of the title
# - ordering (integer) – a number to uniquely identify rows for a given titleId
# - nconst (string) - alphanumeric unique identifier of the name/person
# - category (string) - the category of job that person was in
# - job (string) - the specific job title if applicable, else '\N'
# - characters (string) - the name of the character played if applicable, else '\N'
# 


import pandas as pd
princ = pd.read_csv('../sourceFiles/title.principals.tsv', sep='\t', header=0)
princ.head()


princ = princ[(princ.category.str.contains('act'))][['nconst','tconst']]
actors = actors.merge(princ,how='inner')
print(len(actors))
actors.head()


# ## Download and Process titles
# 
# ***title.akas.tsv.gz - Contains the following information for titles:***
# 
# - titleId (string) - a tconst, an alphanumeric unique identifier of the title
# - ordering (integer) – a number to uniquely identify rows for a given titleId
# - title (string) – the localized title
# - region (string) - the region for this version of the title
# - language (string) - the language of the title
# - types (array) - Enumerated set of attributes for this alternative title. One or more of the following: "alternative", "dvd", "festival", "tv", "video", "working", "original", "imdbDisplay". New values may be added in the future without warning
# - attributes (array) - Additional terms to describe this alternative title, not enumerated
# - isOriginalTitle (boolean) – 0: not original title; 1: original title
# 
# ***Following Filters Applied***
# 
# - Region is US


import pandas as pd
titles = pd.read_csv('../sourceFiles/title.akas.tsv', sep='\t', header=0)
titles = titles[titles.region == 'US']
titles.head()


# ## Download and Process title Basic Info
# 
# ***title.basics.tsv.gz - Contains the following information for titles:***
# 
# - tconst (string) - alphanumeric unique identifier of the title
# - titleType (string) – the type/format of the title (e.g. movie, short, tvseries, tvepisode, video, etc)
# - primaryTitle (string) – the more popular title / the title used by the filmmakers on promotional materials at the point of release
# - originalTitle (string) - original title, in the original language
# - isAdult (boolean) - 0: non-adult title; 1: adult title
# - startYear (YYYY) – represents the release year of a title. In the case of TV Series, it is the series start year
# - endYear (YYYY) – TV Series end year. ‘\N’ for all other title types
# - runtimeMinutes – primary runtime of the title, in minutes
# - genres (string array) – includes up to three genres associated with the title
# 
# ***Following Filters Applied***
# 
# - Title type is movie

import pandas as pd
basics = pd.read_csv('../sourceFiles/title.basics.tsv', sep='\t', header=0)
basics = basics[basics.titleType == 'movie'][['tconst']]
basics.columns = ['titleId']
basics.head()


titles = titles.merge(basics,how='inner')
del basics
print(len(titles))
titles.head()


df_titles = pd.DataFrame((titles[titles.region == 'US']['titleId'].unique()))
del titles
df_titles.columns = ['tconst']
df_titles.head()


# ## Download and Process Ratings Info
# 
# ***title.ratings.tsv.gz – Contains the IMDb rating and votes information for titles***
# 
# - tconst (string) - alphanumeric unique identifier of the title
# - averageRating – weighted average of all the individual user ratings
# - numVotes - number of votes the title has received
# 
# ***Following Filters Applied***
# 
# - numVotes atleast 2000


import pandas as pd
ratings = pd.read_csv('../sourceFiles/title.ratings.tsv', sep='\t', header=0)
print(len(ratings))
ratings.head()


ratings = ratings[ratings.numVotes > 2000]
print(len(ratings))
ratings = ratings[['tconst']]
df_titles = df_titles.merge(ratings,how='inner')
print(len(df_titles))
df_titles.head()


actors = actors.merge(df_titles,how='inner')
actors = actors.reset_index(drop=True)
print(len(actors))
actors.head()             

print("Total Movies : ",len(actors['tconst'].unique()))
print("Total Actors : ",len(actors['nconst'].unique()))


##actors = actors[:100]

# # Write File

actors.to_csv('../files/actorsOrig.csv',header=True,index=False)
print(len(actors))

