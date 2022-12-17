#%%
'''
File name: evolution.py
Author: Loïc Fischer
Date created:16/12/2022
Date last modified: 16/11/2022
Python Version: 3.9.13
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
import sys
import json
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%%
PATH                = 'D:/Ada/'
DATA_CHANNEL        = 'df_channels_en.tsv.gz'
DATA_VIDEO_META     = 'yt_metadata_en.jsonl.gz'
DATA_COMMENT        = 'youtube_comments.tsv.gz'
NUM_COMMENT         = 'num_comments.tsv.gz'
COMPRESSION         = 'gzip'
SEPARATOR           = '\t'

df_channels   = pd.DataFrame()
df_video_meta = pd.DataFrame()
df_comments   = pd.DataFrame()

# Read data from memory and clean it
df_channels = pd.read_csv(os.path.join(PATH,DATA_CHANNEL), compression=COMPRESSION,sep=SEPARATOR)
df_channels = df_channels.dropna()

# Read data from memory and clean it
df_video_meta = pd.read_json(os.path.join(PATH,DATA_VIDEO_META), compression=COMPRESSION,lines=True, nrows=10000)
df_video_meta = df_video_meta.dropna()

# Read data from memory and clean it
df_comments = pd.read_csv(os.path.join(PATH,DATA_COMMENT),sep=SEPARATOR,nrows=1e6)
df_comments = df_comments.dropna()

#%%
channels_cat=list(df_channels.category_cc.unique())
channels_cat.remove('Film and Animation')
channels_cat.append('Film & Animation')
years=list(range(2005,2020))
cross_tab=pd.DataFrame(np.zeros((len(years),len(channels_cat))),years,channels_cat)
#cross_tab=pd.crosstab(years,channels_cat)
#for col in cross_tab.columns:
#    cross_tab[col].values[:] = 0
cross_tab
# %%
'''
i=0
cross_tab_it=cross_tab
chunks = pd.read_json(os.path.join(PATH,DATA_VIDEO_META),compression=COMPRESSION, lines=True, chunksize = 1e5)
for chunk in chunks:
    cross_tab_old=cross_tab_it
    chunk=chunk.drop(['channel_id','crawl_date','description','dislike_count','duration','title','like_count','tags'],axis=1)
    chunk['upload_date']=pd.to_datetime(chunk['upload_date'])
    cross_tab_new=pd.crosstab(chunk.upload_date.dt.year,chunk.categories)
    #a=chunk.groupby(by=[chunk.upload_date.dt.year,chunk.categories])
    #a.size()
    #cross_tab_new
    cross_tab_it=cross_tab_old.add(cross_tab_new,fill_value=0)
    i=i+1
    print(i)
    #if i >9 :
    #    break
print(view_count)
cross_tab_it
cross_tab_it.to_csv(cat_evolution.csv)
'''
# %%

evolution=pd.DataFrame
evolution=pd.read_csv('data/evolution.csv',index_col=0)
evolution=evolution.fillna(0)
evolution=evolution.sort_values(by = 2019, axis = 1,ascending = False)
#evolution=evolution.drop(2019,axis=0)
#evolution.set_index(drop=True,inplace=True)

# %%

sns.set_style("whitegrid")
plt.figure();
ax=sns.lineplot(data=evolution)
ax=sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
ax=plt.ylabel('Number of videos uploaded');
ax=plt.xlabel('Year');
ax=plt.title('Evolution of categories');
ax=plt.savefig('pictures/category_evolution.png',bbox_inches='tight',dpi=150)
plt.show()
#%%
fig = px.line(evolution,title='Evolution of categories',
            labels={"index": "Year",
                    "value": "Number of videos uploaded",
                    "variable":"Category"
                    }
            )
fig.show()
# %%
