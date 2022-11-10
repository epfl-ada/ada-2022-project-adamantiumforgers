# %% 1

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

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

## Create channel_num for all channels in News & Politics

channels = pd.read_csv(PATH_CHANNELS, sep="\t", usecols = ['category_cc','channel'],compression="infer")
channels = channels[channels['category_cc']=='News & Politics'].reset_index(drop=True)
channels = channels['channel'].reset_index()
channels.columns=['channel_num','channel_id']
channels.to_csv("data/channels.csv",sep=";",index=False)
#display(channels)


# %% 4

#Create channel_num for all medias exctracted from AllTimes

channels = pd.read_csv("data/channels.csv", sep=";")
medias = pd.read_csv("LF/channels_w_pol_orr.csv", sep=",", usecols = ['channel','name_cc'])

medias = pd.merge(left=channels, right=medias, how='right', left_on='channel_id', right_on='channel')
medias=medias[['channel_num','channel','name_cc']]
medias.columns=['channel_num','channel_id','name_cc']

for i in range(len(medias)):
    if medias['channel_num'].isnull()[i]==True:
        medias['channel_num'][i]=min(min(medias['channel_num'])-1,-1)

#display(medias)
medias.to_csv("data/medias.csv",sep=";",index=False)


# %% 5

## Create a table display_id -> channel_num of News and Politics

%timeit

# Should we use channels or media ? Change the file here
use = 'channels'

if use =='medias':
    df_channels = pd.read_csv("data/medias.csv", sep=";")
    PATH_RESULT = "data/display_id_to_medias.csv"
else:
    df_channels = pd.read_csv("data/channels.csv", sep=";")
    PATH_RESULT = "data/display_id_to_channels.csv"

display_id_to_channels = pd.DataFrame(columns=['display_id','channel_id','channel_num'])
display_id_to_channels.to_csv("data/display_id_to_channels.csv",sep=";",header=True, index=False)

#for chunk in pd.read_json(PATH_METADATA, lines=True, compression="infer", chunksize=100000):
for chunk in pd.read_json(PATH_METADATA, lines=True, compression="infer", chunksize=10000, nrows=1000000):
    chunk = chunk[chunk.categories == 'News & Politics']
    df_temp = chunk[['display_id','channel_id']]
    df_temp = df_temp.merge(df_channels, on='channel_id',how='inner') # Here, are deleted videos with category News&Politics in a channel whose category is not News&Politics
    df_temp.to_csv(PATH_RESULT,sep=";",mode='a',header=False, index=False)

    #display_id_to_channels = pd.concat([display_id_to_channels,df_temp])

#display_id_to_channels = display_id_to_channels[['display_id','channel_num']]
#display_id_to_channels.columns=['video_id','channel_num']
#display(display_id_to_channels)


# %% 6

# Create a table author -> channel_num

%timeit

# Should we use channels or media ? Change the file here
use = 'channels'

if use =='medias':
    display_id_to_channels = pd.read_csv("data/display_id_to_medias.csv",sep=";",usecols=['display_id','channel_num'])
    PATH_RESULT = "data/authors_to_medias.csv"
else:
    display_id_to_channels = pd.read_csv("data/display_id_to_channels.csv",sep=";",usecols=['display_id','channel_num'])
    PATH_RESULT = "data/authors_to_channels.csv"

display_id_to_channels.columns=['video_id','channel_num']

author_to_channel = pd.DataFrame(columns=['author','channel_num'])
author_to_channel.to_csv(PATH_RESULT,sep=";",header=True, index=False)

for chunk in pd.read_csv(PATH_COMMENTS,sep='\t',usecols = ['author','video_id'],chunksize=100000, nrows=10000000):
    df_temp = chunk.merge(display_id_to_channels, on='video_id')
    df_temp=df_temp[['author','channel_num']]
    # Regroup for this chunk all comments from same authors to same channels (reduces size in memory)
    df_temp= df_temp.groupby(by=['author','channel_num'],as_index =False).size()[['author','channel_num']]
    df_temp.to_csv(PATH_RESULT,sep=";",mode='a',header=False, index=False)

    # As long as it can fit into memory
    #author_to_channel = pd.concat([author_to_channel,df_temp])

# As long as it can fit into memory
#author_to_channel = author_to_channel.reset_index(drop=True)
#display(author_to_channel)



# %% 7

# Create the relational graph - Based on merged between two copies of author_to_channels/medias

%timeit

# Should we use channels or media ? Change the file here
use = 'channels'

if use =='channels':
    PATH_DATA = "data/authors_to_channels.csv"
elif use=='medias':
    PATH_DATA = "data/authors_to_medias.csv"
else:
    PATH_DATA = "data/authors_to_channels_test.csv"


graph = pd.DataFrame(columns=['source','target','weight'])
graph.to_csv("data/graph.csv",sep=";",header=True, index=False)

for chunk in pd.read_csv(PATH_DATA,sep=';',chunksize=100000, nrows=1000):
    chunk=chunk.merge(chunk, on='author', how='inner')
    chunk=chunk[['channel_num_x','channel_num_y']]
    chunk=chunk.groupby(by=['channel_num_x','channel_num_y']).size().reset_index()
    chunk.to_csv("data/graph.csv",sep=";",mode='a',header=False, index=False)
    
    # As long as it can fit into memory
    #graph = pd.concat([graph, merged])

#graph = graph.reset_index(drop=True)
#display(graph)
# %% 8 

# Transform graph to remove directionality and selft pointing edges

edges = pd.read_csv("data/graph.csv",sep=";")

G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight')
G.remove_edges_from(nx.selfloop_edges(G))

edges = nx.to_pandas_edgelist(G)
edges.to_csv("data/graph.csv",sep=";")

display(edges)

# %%

# %%