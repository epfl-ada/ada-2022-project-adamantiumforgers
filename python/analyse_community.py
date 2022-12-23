#%%
'''
File name: news_channel.py
Author: Loïc Fischer
Date created: 05/11/2022
Date last modified: 18/11/2022
Python Version: 3.9.13
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Display of the imported datasets
df_video_meta.head()
df_channels.head(5)
df_comments.head(5)

#%%

news_channels=pd.DataFrame()
news_channels=df_channels[df_channels['category_cc']=='News & Politics']
news_channels=news_channels.drop(['join_date','subscriber_rank_sb','weights'],axis=1)

df_channels_id=pd.DataFrame()
df_channels_id = pd.read_csv('data/channels.csv',sep=';')

news_channels = pd.merge(left=df_channels_id, right=news_channels, left_on='channel_id', right_on='channel')
news_channels=news_channels.drop(['channel'],axis=1)

#%%
communities=pd.read_csv("csv_outputs/louvain_communities_channels_large.csv",sep=";")
communities

news_channels_com = pd.merge(left=news_channels, right=communities, left_on='channel_num', right_on='channel',suffixes=('', ''))
news_channels_com = news_channels_com.drop(['channel'],axis=1)
news_channels_com

news_channels_com
# %%
sns.scatterplot(news_channels_com.community,y=news_channels_com.subscribers_cc,size=news_channels_com.subscribers_cc)

#%%
channel_orr_pol=pd.read_csv('allsides_scraping/csv/channels_yt_all.csv',index_col=0)
channel_orr_pol=channel_orr_pol[['channel','orrientation','confidence']]
merge= pd.merge(left=news_channels_com, right=channel_orr_pol, left_on='channel_id',right_on='channel')
merge=merge.drop(['channel'],axis=1)
# %%
sns.scatterplot(x=merge.orrientation,y=merge.community,hue=merge.community)
# %%
plt.scatter(x=merge.orrientation,y=merge.community)

# %%
import plotly.express as px


fig = px.scatter(merge, y="community", x="orrientation",category_orders={'orrientation':['Left','Lean Left','Center','Lean Right','Right']})

fig.show()
# %%
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter(
    x=merge.orrientation, y=merge.community,
    mode='markers',
    marker_size=list(range(1,52)))
])

fig.show()


# %%
fig = px.imshow(a, text_auto=True,origin='lower',color_continuous_scale='Blues')
fig.show()
fig.write_html('heatmap_comunity.html')
# %%
