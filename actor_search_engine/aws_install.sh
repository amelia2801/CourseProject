Launch Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-083f68207d3376798 (64-bit x86) / ami-08e45d0d0a04bc682 (64-bit Arm)

sudo apt-get update

## Install Python via Anaconda3
###############################
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-5.0.1-Linux-x86_64.sh
sudo reboot

### Install Requirements
sudo apt-get install python3-bs4
conda install selenium
sudo apt-get install chromium-chromedriver

pip install metapy
pip install pytoml
pip install requests
pip install flask

#### Clone and Create idx
git clone https://github.com/aravindsp/CS410_Project
cd CS410_Project/actor_search_engine/
python buildSearchIndex.py

### Start web server
nohup python bgservice.py &