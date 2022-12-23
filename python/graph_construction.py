'''
File name: graph_creation.ipynb
Author: Camille Bernelin
Date created: 10/11/2022
Date last modified: 18/11/2022
Python Version: 3.9.13
'''

# %% 1

import numpy as np
import pandas as pd
import seaborn as sns

# Package for graph management and treatment
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

# Filter for News & Politics channels
channels = channels[channels['category_cc']=='News & Politics'].reset_index(drop=True)
# Create channel numbers out of old indexes
channels = channels['channel'].reset_index()
channels.columns=['channel_num','channel_id']

channels.to_csv("csv_outputs/channels.csv",sep=";",index=False)
channels.head()


# %% 4

#Create channel_num for all medias exctracted from AllTimes

# Load already created channel numbers
channels = pd.read_csv("csv_outputs/channels.csv", sep=";")

medias = pd.read_csv("allsides_scraping/csv/channels_w_pol_orr.csv", sep=",", usecols = ['channel','name_cc'])

medias = pd.merge(left=channels, right=medias, how='right', left_on='channel_id', right_on='channel')
medias=medias[['channel_num','channel','name_cc']]
medias.columns=['channel_num','channel_id','name_cc']

# If the media has no channel number yet, create one
for i in range(len(medias)):
    if medias['channel_num'].isnull()[i]==True:
        medias['channel_num'][i]=min(min(medias['channel_num'])-1,-1) # The created channel number is negative to differentiate with previoulsy created channel numbers
medias = medias.sort_values(by='channel_num')

medias.to_csv("csv_outputs/medias.csv",sep=";",index=False)
medias.head()


# %% 5

## Create a table display_id -> channel_num

# Should we use all N&P channels or just AllSides medias ? Select here medias/channels
use = 'channels'

if use =='medias':
    PATH_DATA = "csv_outputs/medias.csv"
    PATH_RESULT = "csv_outputs/display_id_to_medias.csv" #_temp to avoid overwriting the long one we already created
else:
    PATH_DATA = "csv_outputs/channels.csv"
    PATH_RESULT = "csv_outputs/display_id_to_channels.csv" #_temp to avoid overwriting the long one we already created

# Open the list of channels selected above
df_channels = pd.read_csv(PATH_DATA, sep=";")

# Format output file 
display_id_to_channels = pd.DataFrame(columns=['display_id','channel_id','channel_num'])
display_id_to_channels.to_csv(PATH_RESULT,sep=";",header=True, index=False)

for chunk in pd.read_json(PATH_METADATA, lines=True, compression="infer", chunksize=10000):
    chunk = chunk[chunk.categories == 'News & Politics']
    # Merge with list of News & Politics channels
    chunk = chunk.merge(df_channels, on='channel_id',how='inner') # Here are deleted videos with category News&Politics that are in a channel whose category is not News&Politics
    chunk = chunk[['display_id','channel_id','channel_num']]
    chunk.to_csv(PATH_RESULT,sep=";",mode='a',header=False, index=False)

# Display the result
display_id_to_channels = pd.read_csv(PATH_RESULT,sep=";", nrows=5)
display_id_to_channels.head()


# %% 6

# Create a table author -> channel_num

# Create a table author -> channel_num

# Should we use all N&P channels or just AllSides medias ? Select here medias/channels
use = 'channels'

if use =='medias':
    PATH_DATA = "csv_outputs/display_id_to_medias.csv"
    PATH_RESULT = "csv_outputs/authors_to_medias_temp.csv" #_temp to avoid overwriting the long one we already created

else:
    PATH_DATA = "csv_outputs/display_id_to_channels.csv"
    PATH_RESULT = "csv_outputs/authors_to_channels.csv"#_temp to avoid overwriting the long one we already created

# Open table with channels selected above
display_id_to_channels = pd.read_csv(PATH_DATA,sep=";",usecols=['display_id','channel_num'])
display_id_to_channels.columns=['video_id','channel_num']

# Format output file 
author_to_channel = pd.DataFrame(columns=['author','channel_num'])
author_to_channel.to_csv(PATH_RESULT,sep=";",header=True, index=False)

for chunk in pd.read_csv(PATH_COMMENTS,sep='\t',usecols = ['author','video_id'],chunksize=1000000, nrows=100000000):
    df_temp = chunk.merge(display_id_to_channels, on='video_id')
    df_temp=df_temp[['author','channel_num']]
    # Regroup for this chunk all comments from same authors to same channels (reduces size in memory)
    df_temp = df_temp.groupby(by=['author','channel_num'],as_index =False).size()
    # Disregard the size column, as we only add a line for the first comment of the author to the channel
    df_temp = df_temp[['author','channel_num']]
    df_temp.to_csv(PATH_RESULT,sep=";",mode='a',header=False, index=False)

# Display the result
author_to_channel = pd.read_csv(PATH_RESULT,sep=";")
author_to_channel.head()

# %% 7

# Create the relational graph

# Should we use all N&P channels or just AllSides medias ? Select here medias/channels or test
use = 'medias'

if use =='channels':
    PATH_DATA = "csv_outputs/authors_to_channels.csv"
elif use=='medias':
    PATH_DATA = "csv_outputs/authors_to_medias.csv"
else:
    PATH_DATA = "csv_outputs/authors_to_channels_test.csv"

PATH_RESULT = "csv_outputs/graph.csv"

# Format output file
graph = pd.DataFrame(columns=['source','target','weight'])
graph.to_csv(PATH_RESULT,sep=";",header=True, index=False)

for chunk in pd.read_csv(PATH_DATA,sep=';',chunksize=100000):
    # Merge the chunk with itself to obtain all combinations
    chunk=chunk.merge(chunk, on='author', how='inner')
    chunk=chunk[['channel_num_x','channel_num_y']]
    chunk=chunk.groupby(by=['channel_num_x','channel_num_y']).size().reset_index()
    chunk.to_csv(PATH_RESULT,sep=";",mode='a',header=False, index=False)

# Display result
graph = pd.read_csv(PATH_RESULT, sep=";")
graph.head()

# %% 8 

# Transform graph to remove directionality and selft pointing edges

# Load previously created graph
edges = pd.read_csv("csv_outputs/graph.csv",sep=";")

# Created an undirected graph from the edges list
G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.Graph())

# Remove self pointing edges
G.remove_edges_from(nx.selfloop_edges(G))

edges = nx.to_pandas_edgelist(G)
edges.to_csv("csv_outputs/graph.csv",sep=";",index=False)

# Display result
edges.head()

# %% 9

# WORK IN PROGRESS....

# Try to find correlation between communities identified and channels political orientation

# Exctract from Gephi or clustering file a modularity file giving the community numbers
modularity = pd.read_csv("modularity.csv",sep=',',usecols=['Id','modularity_class'])

medias = pd.read_csv("csv_outputs/medias.csv",sep=';', usecols=['channel_num','name_cc'])
politics = pd.read_csv("allsides_scraping/csv/channels_w_pol_orr.csv",sep=',',usecols=['name_cc','orrientation'])

merged = modularity.merge(medias, right_on='channel_num',left_on='Id')
merged = merged.merge(politics, on='name_cc')

# Give a score to political orientation
def orientation(a):
    if a=='Lean Right':
        return 0
    if a=='Right':
        return 1
    if a=='Center':
        return 2
    if a=='Left':
        return 3
    if a=='Lean Left':
        return 4

merged.insert(2,column='orientation',value=merged['orrientation'].apply(orientation))
merged=merged[['modularity_class','orientation']]
merged.corr()

merged.to_csv("csv_outputs/modularity_to_politics.csv",sep=";")




# %%