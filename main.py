# %% [markdown]
# # YouNiverse Project
# By **AdamantiumForgers**
# 
# *Study of the politic polarization in various topics (based on big media followers) : Analyze vocabulary in videos posted (Title and description), How the videos are reveived (likes, dislikes, views),list of user_id and their behaviour in the platform, coverage of events (Elections, war)*
# 
# # Dataset
# YouNiverse comprises metadata from over 136k channels and 72.9M videos (English-Speaking only!) published between May 2005 and October 2019, as well as channel-level time-series data with weekly subscriber and view counts.
# 
# Source: https://zenodo.org/record/4650046 \
# Github: https://github.com/epfl-dlab/YouNiverse \
# Size: 111 GB (compressed, in total)
# 
# **List of files**
# * df_channels_en.tsv.gz (6MB): List of the 136'471 channels with some infos (state in october 2019)
# * df_timeseries_en.tsv.gz (571MB): Weekly timeseries for each channel, from 03 July 2017 to 23 October 2019
# * num_comments.tsv.gz (755MB): List of videos (display_id) with their number of comments
# * num_comments_authors.tsv.gz (1.4GB):  ?????????????????????
# * youtube_comments.tsv.gz (77.2GB): ~8.6B comments made by ~449M users in 20.5M videos. Each rows = 1 comment: user id, a video id, number of replies and likes the comment received
# * yt_metadata_en.jsonl.gz (13.6GB): metadata data related to ~73M videos from ~137k channels
# * yt_metadata_helper.feather (2.8GB): Same as jsonl except description, tags, and title (the largest fields)

# %% [markdown]
# # Packages

# %%
import numpy as np
import pandas as pd
import seaborn as sns

import os
import json
#import glob
#import gzip
#import swifter
#import langdetect
#import zstandard as zstd
#import matplotlib as mpl
#import scipy.stats as stats
#import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
#from matplotlib.lines import Line2D
#import matplotlib.font_manager as font_manager

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'



# %%
#class zreader:
#
#    def __init__(self, file, chunk_size=16384):
#        self.fh = open(file, 'rb')
#        self.chunk_size = chunk_size
#        self.dctx = zstd.ZstdDecompressor()
#        self.reader = self.dctx.stream_reader(self.fh)
#        self.buffer = ''
#
#    def readlines(self):
#        while True:
#            chunk = self.reader.read(self.chunk_size).decode("utf-8", errors="ignore")
#            if not chunk:
#                break
#            lines = (self.buffer + chunk).split("\n")
#
#            for line in lines[:-1]:
#                yield line
#
#            self.buffer = lines[-1]

#reader = Zreader("PATH_COMMENTS", chunk_size=16384)

# %% [markdown]
# ## Paths

# %%
## Directories for the data.
## Large data can be stored on an external drive and accessed by creating a simlink to a "large" directory in the data folder
## ln -s target_path link_path

DIR = "data/"
DIR_LARGE = "data/large/"

## Path for each file
PATH_TIME_SERIES = DIR + "df_timeseries_en.tsv.gz"
PATH_CHANNELS = DIR + "df_channels_en.tsv.gz"
PATH_NUM_COMMENTS = DIR + "num_comments.tsv.gz"
PATH_NUM_COMMENTS_AUTHORS = DIR + "num_comments_authors.tsv.gz"
PATH_METADATA = DIR_LARGE + "yt_metadata_en.jsonl.gz"
PATH_METADATA_HELPER = DIR + "yt_metadata_helper.feather"
PATH_COMMENTS = DIR_LARGE + "youtube_comments.tsv.gz"


# %% [markdown]
# ## Load data


# %%
## Can be read entirely 
df_channels = pd.read_csv(PATH_CHANNELS, compression="infer", sep="\t", nrows=10000)
df_timeseries = pd.read_csv(PATH_TIME_SERIES, compression="infer", sep="\t", nrows=10000) # Bigger, but can still be read entirely

## Feather files not useful (use the whole json file instead, it's simpler since read_feather does not have any nrows or chunksize options...)
#df_metadata_light = pd.read_feather(PATH_METADATA_HELPER, use_threads=False, columns=["channel_id", "categories", "dislike_count", "like_count", "duration", "upload_date", "view_count"])
#df_metadata_light = pd.read_feather(PATH_METADATA_HELPER,[1,5]) #Only 1:channel_id and 5:like_count

## Too big, only read nrows then apply by chunks
df_comments = pd.read_csv(PATH_COMMENTS, compression="infer", sep="\t", nrows=10000)
df_num_comments = pd.read_csv(PATH_NUM_COMMENTS, compression="infer", sep="\t", nrows=10000)
df_num_comments_authors = pd.read_csv(PATH_NUM_COMMENTS_AUTHORS, compression="infer", sep="\t", nrows=10000)
df_metadata = pd.read_json(PATH_METADATA, compression='infer', lines=True, nrows=10000)

## Preprocess date fields
df_channels["join_date"] = pd.to_datetime(df_channels["join_date"])
df_timeseries["datetime"] = pd.to_datetime(df_timeseries["datetime"])


# %% [markdown]
# ## Work with the data


# %%
## Create function on the dataframe
def do_stuff(df):
    # For example, filter all channels with more than 10M subs
    #return df[df['subscribers_cc'] > 1e7]
    return df.iloc[1:2]

## Then apply to the whole dataset by a loop and using the argument "chunksize"
result = None  
df = pd.read_csv(PATH_CHANNELS, compression="infer", sep="\t", chunksize=10000)
for chunk in df:
    df_temp = do_stuff(chunk)
    result = pd.concat([result, df_temp])
result.head(5)
print(len(result))


#%%
## Read json file by chunks
## Output is too big to be kept in memory directly -> export to csv at each iteration
filtered_metadata = pd.DataFrame()
df = pd.read_json(PATH_METADATA, lines=True, compression="infer", chunksize=1e5, nrows=1e7) 
#Keep chunksize <1e6 to limit memory usage (<5GB). Total size ~70e7
for i, chunk in enumerate(df):
    df_temp = do_stuff(chunk)
    
    ## If the result is not too big
    #filtered_metadata = pd.concat([filtered_metadata, df_temp])

    ## Otherwise, write by chunk in a csv file
    ## First chunk: writing mode and add header. Then, append mode and do not add headers
    mode = 'w' if i == 0 else 'a'
    header = (i == 0)
    print(f"Progression: Chunk number {i} / {1e7/1e5}")
    ## Save (compressed or not)
    df_temp.to_csv("data/filtered_metadata.csv", columns=df_temp.columns, index=False, header=header, mode=mode)
    df_temp.to_csv("data/filtered_metadata.csv.gz", columns=df_temp.columns, index=False, header=header, mode=mode,compression='gzip')


#display(filtered_metadata)
#len(filtered_metadata)


# %% [markdown]
# ## Overview of all dataframes


# %%
df_channels.head()
df_channels[df_channels['subscribers_cc'] > 1e7]



# %%
df_comments.head()


# %%
df_num_comments.head()


# %%
df_num_comments_authors.head()


# %%
df_metadata.head(3)


# %%



