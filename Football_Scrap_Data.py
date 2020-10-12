import numpy as np# linear algebra 
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv) 
import requests 
from bs4 import BeautifulSoup


# In[ ]:


# create urls for all seasons of all leagues 
base_url = 'https://understat.com/league' 
leagues = ['La_liga', 'EPL', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL'] 
seasons = ['2014', '2015', '2016', '2017', '2018','2019','2020']


# Starting with latest data for Spanish league, because I'm a Barcelona fan 
url = base_url+'/'+leagues[1]+'/'+seasons[5] 
res = requests.get(url) 
soup = BeautifulSoup(res.content, "lxml")
# Based on the structure of the webpage, I found that data is in the JSON variable, under 'script' tags 
scripts = soup.find_all('script')


# In[ ]:


scripts


# In[ ]:


## After creating a soup of html tags, it becomes a string, so we just find the text and extract the JSON from it
import json 

string_with_json_obj = '' 

# Find data for teams 
for el in scripts: 
    if 'teamsData' in el.text: 
        string_with_json_obj = el.text.strip()
        
## This just strips out the whitespace
        
print(string_with_json_obj)

# strip unnecessary symbols and get only JSON data 
ind_start = string_with_json_obj.index("('")+2  # Get first index
ind_end = string_with_json_obj.index("')")  # Get last index
json_data = string_with_json_obj[ind_start:ind_end] 
json_data = json_data.encode('utf8').decode('unicode_escape')




df = pd.read_json (json_data)
print (df)


# In[ ]:


# Get teams and their relevant ids and put them into separate dictionary 
data = json.loads(json_data)
teams = {} 
for id in data.keys(): 
    teams[id] = data[id]['title'] ##This returns the team names


# In[ ]:


## the json object is made of 3 main keys - id, title and history. The first layer of dictionary uses id as well
data


# In[ ]:


## Now, we see that the history tab is where the data is, lets create a dframe of the data
metrics = []
values = []

for id in data.keys():
    metrics = list(data[id]['history'][0].keys()) 
    values = list(data[id]['history'][0].values()) 


# In[ ]:


df = pd.DataFrame(columns = ['metrics','values'])


# In[ ]:


df['metrics'],df['values']=metrics,values


# In[ ]:


df


# In[ ]:


teams


# In[ ]:


arsenal_data = [] 
for row in data['83']['history']:
    arsenal_data.append(list(row.values())) 
df_arsenal = pd.DataFrame(arsenal_data, columns=metrics)
df_arsenal.head(2)


# In[ ]:


df_arsenal


# In[ ]:


import seaborn as sns
sns.factorplot('xG',data=df_arsenal)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


data=df_arsenal.where(df_arsenal["h_a"]=="h")
data


# In[ ]:


sns.factorplot(y='xG', x = 'date',data=df_arsenal.where(df_arsenal["h_a"]=="h"),hue='h_a')
sns.factorplot(y='xG', x = 'date',data=df_arsenal.where(df_arsenal["h_a"]=="a"),hue='h_a')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 6))
sns.boxplot(y="xG", data=df_arsenal.where(df_arsenal["h_a"]=="h"), ax = ax1)
ax1.set_xlabel("Home");
ax1.set_ylabel("xG");

sns.boxplot(y="xG", data = df_arsenal.where(df_arsenal["h_a"]=="a"), ax = ax2)

ax2.set_xlabel("Away");
ax2.set_ylabel("xG");
