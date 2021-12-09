# Custom Search Engine User Interface

- Click on following link
- Enter text to search. Eg : "Time Travel"

http://54.176.35.246:8080/

# Software Usage Tutorial Presentation:
https://drive.google.com/file/d/1P6hGsJ0Lwrk1OmJexMmp0VXBKdpC-GhF/view?usp=sharing

# Custom Search Engine User Interface

http://54.176.35.246:8080/

# Execute web server Locally (Mac OS)

To Execute Locally :

- Clone the Repo
- Go to directory actor_search_engine
- Execute buildSearchIndex.py or buildSearchIndex.ipynb to create inverted index - idx folder
- python movieApp.py

Output :
 * Serving Flask app "movieApp" (lazy loading)
 * Environment: production

 * Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)
 
 # Execute web server on the cloud (AWS)
 
 ### Launch Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-083f68207d3376798 (64-bit x86)
 sudo apt-get update
 
### Install Python via Anaconda3
- sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
- wget https://repo.anaconda.com/archive/Anaconda3-5.0.1-Linux-x86_64.sh
- sudo reboot

### Install Requirements
- sudo apt-get install python3-bs4
- conda install selenium
- sudo apt-get install chromium-chromedriver

- pip install --upgrade pandas
- pip install metapy
- pip install pytoml
- pip install requests
- pip install flask

### Clone and Create idx
- clone the repo
- cd CS410_Project/actor_search_engine/
- python buildSearchIndex.py

### Start web server
nohup python movieApp.py &

 
# Run all processes from scratch (Mac OS)
 
 
***Steps***

## Step 1 : Clone the Repo
Go to directory actor_search_engine

## Step 2 : Download following files and extract the tsv files to "SourceFiles" directory
Link - https://datasets.imdbws.com/

Files :

- name.basics.tsv.gz
- title.akas.tsv.gz
- title.basics.tsv.gz
- title.principals.tsv.gz
- title.ratings.tsv.gz

## Step 3 : Scrape files and build corpus
 
 There are 2 options. Execute either one of them
 
 ### Option 1 - Execute notebooks in following order
 
 1) prepareMovieActorsFile
 2) ScrapeMoviesActors.ipynb
 3) ScrapeMoviesTags.ipynb
 4) buildCorpus.ipynb
 
 OR 
 
 ### Option 2 - Execute shell script 
 executeAllScripts.sh
 
 
## Step 4 : Build Inverted Index

Execute buildSearchIndex

## Step 5 : Render web page

Execute "python movieApp.py"
