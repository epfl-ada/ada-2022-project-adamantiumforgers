# %% 1

import numpy as np
import pandas as pd
import seaborn as sns

import os
import json

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from os.path import exists

# %% 2

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

# %% 3

## Create a table channel_num -> channel of News and Politics

channels = pd.read_csv(PATH_CHANNELS, sep="\t", usecols = ['category_cc','channel'],compression="infer")
channels = channels[channels['category_cc']=='News & Politics'].reset_index(drop=True)
channels = channels['channel'].reset_index()
channels.columns=['channel_num','channel_id']
channels.to_csv("data/channels.csv",sep=";",index=False)
display(channels)

# %% 4

## Create a table display_id -> channel_num of News and Politics

display_id_to_channels = pd.DataFrame(columns=['display_id','channel_id','channel_num'])
display_id_to_channels.to_csv("data/display_id_to_channels.csv",sep=";",header=True, index=False)

for chunk in pd.read_json(PATH_METADATA, lines=True, compression="infer", chunksize=100000):
#for chunk in pd.read_json(PATH_METADATA, lines=True, compression="infer", chunksize=10000, nrows=10000):
    chunk = chunk[chunk.categories == 'News & Politics']
    df_temp = chunk[['display_id','channel_id']]
    df_temp = df_temp.merge(channels, on='channel_id',how='inner') # Here, are deleted videos with category News&Politics in a channel whose category is not News&Politics
    df_temp.to_csv("data/display_id_to_channels.csv",sep=";",mode='a',header=False, index=False)

    #display_id_to_channels = pd.concat([display_id_to_channels,df_temp])

#display_id_to_channels = display_id_to_channels[['display_id','channel_num']]
#display_id_to_channels.columns=['video_id','channel_num']
#display(display_id_to_channels)

# %% 5

# Create a table author -> channel_num

display_id_to_channels = pd.read_csv("data/display_id_to_channels.csv",sep=";")
display_id_to_channels = display_id_to_channels[['display_id','channel_num']]
display_id_to_channels.columns=['video_id','channel_num']

author_to_channel = pd.DataFrame(columns=['author','channel_num'])
author_to_channel.to_csv("data/authors_to_channel.csv",sep=";",header=True, index=False)

for chunk in pd.read_csv(PATH_COMMENTS,sep='\t',usecols = ['author','video_id'],chunksize=100000, nrows=100000):
    df_temp = chunk.merge(display_id_to_channels, on='video_id')
    df_temp=df_temp[['author','channel_num']]
    # Regroup for this chunk all comments from same authors to same channels (reduces size in memory)
    df_temp= df_temp.groupby(by=['author','channel_num'],as_index =False).size()[['author','channel_num']]
    df_temp.to_csv("data/authors_to_channel.csv",sep=";",mode='a',header=False, index=False)

    # As long as it can fit into memory
    #author_to_channel = pd.concat([author_to_channel,df_temp])

# As long as it can fit into memory
#author_to_channel = author_to_channel.reset_index(drop=True)
#display(author_to_channel)

# %% 6

# Create the relational graph - Based on groupby channels
graph = pd.DataFrame(columns=['Source','Target','Weight'])
graph.to_csv("data/graph.csv",sep=";",header=True, index=False)

# As long as it can fit into memory
# Group by channels
group_by_channel = author_to_channel.groupby(by='channel_num')
channels = group_by_channel.size().reset_index()['channel_num']

for channel1 in range(0,len(channels),1):
    group1 = group_by_channel.get_group(channels[channel1])
    for channel2 in range(channel1+1,len(channels),1): # Note that none of the channel couples are browsed twice
        group2 = group_by_channel.get_group(channels[channel2])
        # Merge on authors having written to a channel couple
        merged = pd.merge(group1,group2,on='author')
        merged=merged[['channel_num_x','channel_num_y']]
        # Add up all edges linking the same channels
        merged=merged.groupby(by=['channel_num_x','channel_num_y']).size().reset_index()
        merged.columns=['Source','Target','Weight']
        merged.to_csv("data/graph.csv",sep=";",mode='a',header=False, index=False)

        # As long as it can fit into memory
        #graph = pd.concat([graph, merged])

#graph = graph.reset_index(drop=True)
#display(graph)

# %%




# %%
