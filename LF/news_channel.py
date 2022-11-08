#%%
'''
File name: news_channel.py
Author: Loïc Fischer
Date created: 05/11/2022
Date last modified: 05/11/2022
Python Version: 3.8
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

df_video_meta.head()
df_channels.head(5)
df_comments.head(5)



#%%
df_channels_md=pd.DataFrame()
df_media=pd.DataFrame()

df_media=pd.read_csv('media_ft_man_mod.csv')

df_channels['name_cc']=df_channels['name_cc'].str.casefold()
df_media['name']=df_media['name'].str.casefold()

df_channels_md=df_channels[(df_channels['name_cc'].isin(df_media['name']))]
df_channels_md=df_channels_md[df_channels_md['subscribers_cc']>1e4]
df_channels_md=df_channels_md.reset_index(drop=True)


merged_inner = pd.merge(left=df_channels, right=df_media, left_on='name_cc', right_on='name')
merged_inner = merged_inner.drop(['Unnamed: 0','name'], axis=1)
merged_inner = merged_inner.sort_values(by=['name_cc'])
merged_inner = merged_inner.reset_index(drop=True)
merged_inner = merged_inner.drop_duplicates()
#merged_inner = merged_inner[merged_inner['subscribers_cc']>1e6]

with pd.option_context('display.max_rows', None,):
    merged_inner

merged_inner.loc[merged_inner['name_cc'].duplicated(False)]
#%%
## using the method contains

#df3 = pd.merge(left=df_channels, right=df_media, left_on='name_cc', right_on='name')
#df3=df_channels[(df_channels['name_cc'].str.contains('|'.join(df_media['name'])))&(df_channels['category_cc']=='News & Politics')&(df_channels['subscribers_cc']>1e5)]

#'|'.join(df_media['name'])
#%%
## gettint all the channel with string inside

#for ind1 in df_media.index:
#   df_media.loc[ind1, 'name_cc'] = ', '.join(list(df_channels_news[df_channels_news['name_cc'].str.contains(df_media['name'][ind1])]['name_cc']))
#with pd.option_context('display.max_rows', None,):
#    df_media[df_media['name_cc'].str.len() > 0]

#%%



sns.histplot(merged_inner['category_cc'])
plt.xticks(rotation=90);


# %%
sns.histplot(merged_inner['subscribers_cc'],binwidth=1e5)
plt.xticks(rotation=90);
# %%
sns.histplot(merged_inner['orr'],binwidth=1e5)
plt.xticks(rotation=90);
# %%

# %%
def test_channel_name(testword,news=True):
    with pd.option_context('display.max_rows',None,'display.max_columns', None,'display.max_colwidth',4000):
        if news == True:
            return df_channels[(df_channels['name_cc'].str.contains(testword.casefold()))&(df_channels['category_cc']=='News & Politics')]
        else:
            return df_channels[(df_channels['name_cc'].str.contains(testword.casefold()))]
# %%

def test_channel_id(testword):
    with pd.option_context('display.max_rows',None,'display.max_columns', None,'display.max_colwidth',4000):
        return df_channels[(df_channels['channel'].str.contains(testword.casefold()))]

# %%
