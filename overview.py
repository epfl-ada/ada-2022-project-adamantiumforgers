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
import os
import numpy as np
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
# %%
## Pie chart number of channels

palette_color =sns.color_palette('tab20')
g=df_channels.groupby(by=['category_cc'])
a=g.count().sort_values(by='subscribers_cc',ascending=False)
my_labels=list(a['videos_cc'].index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(a['subscribers_cc'],labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);

# %%
## Pie chart number of subscribers
palette_color =sns.color_palette('tab20')
g=df_channels.groupby(by=['category_cc'])
a=g.sum().sort_values(by='subscribers_cc',ascending=False)
my_labels=list(a['subscribers_cc'].index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(a['subscribers_cc'],labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);
# %%
## Pie chart number of videos
palette_color =sns.color_palette('tab20')
g=df_channels.groupby(by=['category_cc'])
a=g.sum().sort_values(by='videos_cc',ascending=False)
my_labels=list(a['videos_cc'].index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(a['videos_cc'],labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);
# %%
df_video_meta.groupby(by='categories')
# %%
df_video_meta.groupby(by='categories').sum()

# %%
'''
i=0
view_count=0
chunks = pd.read_json(os.path.join(PATH,DATA_VIDEO_META),compression=COMPRESSION, lines=True, chunksize = 2e5)
for chunk in chunks:
    view_count_old=view_count
    view_count=chunk.groupby(by='categories').sum('view_count')
    view_count=view_count_old+view_count
    #chunk.groupby(by='categories').sum('view_count')
    i=i+1
    print(i)
print(view_count)
'''
# %%
## Total number of views
view_cat=pd.read_csv('data/view_count.csv')
view_cat=view_cat.dropna()
view_cat=view_cat.sort_values(by='view_count',ascending=False)
my_labels=list(view_cat.categories)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(view_cat.view_count,labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);

# %%
import plotly.express as px
import plotly.graph_objects as go


fig = go.Figure(data=[go.Pie(   labels=my_labels,
                                values=view_cat.view_count,
                                marker_colors=px.colors.qualitative.Set3,
                                pull=my_explode
                                
                                
                                )])
fig.update_layout(
    autosize=False,
    width=750,
    height=500)

fig.show()
# %%
