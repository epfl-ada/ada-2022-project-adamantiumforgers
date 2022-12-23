# %%
'''
File name: news_channel.py
Author: Loïc Fischer
Date created: 05/11/2022
Date last modified: 18/11/2022
Python Version: 3.9.13
'''

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import string
import sklearn
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
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

## Note: only partial data have been imported for a faster processing time.

# %%
# Merging the dataset from Allside and the one from Youniverse

dataset="all" # Choose the dataset : "ft" or "all"

df_channels_md=pd.DataFrame()
df_media=pd.DataFrame()

if dataset=="all":
    df_media=pd.read_csv('csv/media_all_raw.csv',index_col=0)
elif dataset=="ft":
    df_media=pd.read_csv('csv/media_ft_raw.csv',index_col=0)
else:
    print("You need to choose a dataset")

df_channels['name_cc']=df_channels['name_cc'].str.casefold()
df_media['name']=df_media['name'].str.casefold()

#df_channels_md=df_channels[(df_channels['name_cc'].isin(df_media['name']))]
#df_channels_md=df_channels_md[df_channels_md['subscribers_cc']>1e6]
#df_channels_md=df_channels_md.reset_index(drop=True)

df_news_channels=df_channels[df_channels['category_cc']=='News & Politics']

merged_inner = pd.merge(left=df_channels, right=df_media, left_on='name_cc', right_on='name')
merged_inner = merged_inner.drop(['name'], axis=1)
merged_inner = merged_inner.sort_values(by=['name_cc'])
merged_inner = merged_inner.reset_index(drop=True)
merged_inner = merged_inner.drop_duplicates()
#merged_inner = merged_inner[merged_inner['subscribers_cc']>1e6]
merged_inner=merged_inner[merged_inner['category_cc']=='News & Politics']

with pd.option_context('display.max_rows', None,):
    merged_inner

#merged_inner.loc[merged_inner['name_cc'].duplicated(False)]
if dataset=="all":
    merged_inner.to_csv('csv/channels_yt_all_test.csv')
elif dataset=="ft":
    merged_inner.to_csv('csv/channels_yt_ft_test.csv')
else:
    print("You need to choose a dataset")

len(merged_inner)
# %%
'''
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
'''

# %%
#import Levenshtein

#lv_dist=pd.DataFrame(columns=['Levenshtein','name_ch','name_media'])

#for name_1 in df_news_channels.name_cc:
#    for name_2 in df_media.name:
#        if Levenshtein.distance(name_1,name_2)==1:
#            lv_dist=lv_dist.append({'Levenshtein':Levenshtein.distance(name_1,name_2),'name_ch':name_1,'name_media':name_2},ignore_index=True)
#lv_dist

# %%
'''
from sklearn.metrics.pairwise import cosine_similaity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords=stopwords.word('english')
'''
# %%
'''
def clean_string(text):
    text=''.join([word for word in text if word not in string.punctuation])
    text=text.lower()
    text=' '.join([word for word in text.split() if word not in stopwords])

    return text

'''
# %%
## using the method contains

df3 = pd.merge(left=df_channels, right=df_media, left_on='name_cc', right_on='name')
df3=df_channels[(df_channels['name_cc'].str.contains('|'.join(df_media['name'])))&(df_channels['category_cc']=='News & Politics')&(df_channels['subscribers_cc']>1e4)]

'|'.join(df_media['name'])
# %%
## gettint all the channel with string inside

#for ind1 in df_media.index:
#   df_media.loc[ind1, 'name_cc'] = ', '.join(list(df_channels_news[df_channels_news['name_cc'].str.contains(df_media['name'][ind1])]['name_cc']))
#with pd.option_context('display.max_rows', None,):
#    df_media[df_media['name_cc'].str.len() > 0]


# %%


fig, axs = plt.subplots(1,4, figsize=(20,5));

sns.histplot(data=merged_inner, x='category_cc', ax=axs[0]).tick_params('x', labelrotation=90)
sns.histplot(data=merged_inner, x='category_cc', ax=axs[0]).title.set_text("Categories of {} medias channels".format(dataset))

sns.histplot(data=merged_inner, x='subscribers_cc', ax=axs[1]);
sns.histplot(data=merged_inner, x='subscribers_cc', ax=axs[1]).title.set_text("Number of subscribers of {} medias channels".format(dataset));

sns.histplot(data=merged_inner, x='orrientation', ax=axs[2]).tick_params('x', labelrotation=90);
sns.histplot(data=merged_inner, x='orrientation', ax=axs[2]).title.set_text("Orientation of {} medias channels".format(dataset))

sns.histplot(data=merged_inner, x='confidence', ax=axs[3]).tick_params('x', labelrotation=90);
sns.histplot(data=merged_inner, x='confidence', ax=axs[3]).title.set_text("Confidence on the orrientaton of {} medias channels".format(dataset))


plt.tight_layout();
plt.savefig('figures/hist_{}.png'.format(dataset));


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

