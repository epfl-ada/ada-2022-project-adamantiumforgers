#%%
'''
File name: media_search_on_yt.py
Author: Loïc Fischer
Date created: 20/11/2022
Date last modified: 20/11/2022
Python Version: 3.9.13
'''

#%%
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

#%%
PATH                = 'D:/Ada/'
DATA_CHANNEL        = 'df_channels_en.tsv.gz'
DATA_VIDEO_META     = 'yt_metadata_en.jsonl.gz'
DATA_COMMENT        = 'youtube_comments.tsv.gz'
NUM_COMMENT         = 'num_comments.tsv.gz'
COMPRESSION         = 'gzip'
SEPARATOR           = '\t'

df_channels   = pd.DataFrame()


# Read data from memory and clean it
df_channels = pd.read_csv(os.path.join(PATH,DATA_CHANNEL), compression=COMPRESSION,sep=SEPARATOR)
df_channels = df_channels.dropna()


#%%
df_news_channels=df_channels[df_channels['category_cc']=='News & Politics']
df_news_channels['name_cc']=df_news_channels['name_cc'].str.casefold()

# %%
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

# %%

name_ch=df_news_channels.name_cc
name_md=df_media.name


#%%
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

stopwords=stopwords.words('english')

# %%
def clean_string(text):
    text=''.join([word for word in text if word not in string.punctuation])
    text=' '.join([word for word in text.split() if word not in stopwords])
    return text

#%%
lv_dist=pd.DataFrame()
i=1
for name_1 in df_news_channels.name_cc:
    for name_2 in df_media.name:
        name_11=clean_string(name_1)
        name_22=clean_string(name_2)
        #if name_11.isempty():
        #   break
        if name_11==name_22:
            df_news_channels[df_news_channels.name_cc==name_11]
            i=i+1

# %%
