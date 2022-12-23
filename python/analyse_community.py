# %%
'''
File name: analyse_community.py
Author: Loïc Fischer
Date created: 05/11/2022
Date last modified: 18/11/2022
Python Version: 3.9.13

reference ipynp : analyse_community.ipynb

This file analyze the different communities and tries to give them a political orientation from the allside database
'''

# %%
## Import Libraries
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import os
import sys
import json

## Show more than one display per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
## Read the data

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
df_video_meta.head(2)
df_channels.head(2)
df_comments.head(2)

# %%
## Creating dataframe with only news and politics channels
news_channels=pd.DataFrame()
news_channels=df_channels[df_channels['category_cc']=='News & Politics']

## Cleaning
news_channels=news_channels.drop(['join_date','subscriber_rank_sb','weights'],axis=1)

## Dataframe with channel id and community
df_channels_id=pd.DataFrame()
df_channels_id = pd.read_csv('../data/channels.csv',sep=';')

## Merging on channel id
news_channels = pd.merge(left=df_channels_id, right=news_channels, left_on='channel_id', right_on='channel')
news_channels=news_channels.drop(['channel'],axis=1)

news_channels.head()
# %%
## Reading communities
communities=pd.read_csv("../csv_outputs/louvain_communities_channels_large.csv",sep=";")
communities.head()

## Merging ang cleaning
news_channels_com = pd.merge(left=news_channels, right=communities, left_on='channel_num', right_on='channel',suffixes=('', ''))
news_channels_com = news_channels_com.drop(['channel'],axis=1)
news_channels_com.head()


# %%
## Merging with the data from allside
channel_orr_pol=pd.read_csv('../allsides_scraping/csv/channels_yt_all.csv',index_col=0)

## Dropping useless columns
channel_orr_pol=channel_orr_pol[['channel','orrientation','confidence']]

## Merging the allside dataset and the channel one
merge= pd.merge(left=news_channels_com, right=channel_orr_pol, left_on='channel_id',right_on='channel')
merge=merge.drop(['channel'],axis=1)

## Display
merge.head()

# %%

## Creating the list for the hover
merge_wo_dup=merge.drop_duplicates(subset='channel_id' ,keep='last')
a=pd.crosstab(merge_wo_dup.community,merge_wo_dup.orrientation)
extend_list = []
array=[]

for com in a.index:
    for orr in a.columns:
        temp=merge_wo_dup[(merge_wo_dup['orrientation']==orr)&(merge_wo_dup['community']==com)].name_cc.values
        #empty=np.array2string(temp)
        extend_list.append(temp)
array.append(extend_list)


# %%
## Creating the heat map

## Drop duplicates
merge_wo_dup=merge.drop_duplicates(subset='channel_id' ,keep='last')

## Create crosstab
a=pd.crosstab(merge_wo_dup.community,merge_wo_dup.orrientation)

## Reorder the index
a=a.reindex(['Left','Lean Left','Center','Lean Right','Right'],axis=1)

## Usefull for the heatmap
np.random.seed(0)
z3 = np.random.random((6, 5))

#Set the hover data
z2=[["CNN, BuzzFeed News, HuffPost, New York Daily News, The Boston Globe", "ABC News, CBS News, NBC News, USA TODAY, Washington Post, CNN Business, NowThis News, Los Angeles Times, Miami Herald, Chicago Sun-Times, ProPublica, Atlanta Black Star", "The Oregonian, Newsy, Chicago Tribune, Toronto Star, Orlando Sentinel, Education Week", '', "Michelle Malkin"],
["MSNBC, Democracy Now!, The New Yorker, Robert Reich, The Intercept, Hasan Piker, The Nation, Mother Jones, truthdig, Daily Kos",'', "PBS NewsHour, C-SPAN, Newsweek", '', ''],
['', "The Independent", "Financial Times", "spiked", ''],
['', "GLOBAL News", "Honolulu Civil Beat", '', "Project Veritas,Church Militant"],
['', '', "Tim Pool", '', ''],
['', '', "Roll Call", "John Stossel, National Post", "The Daily Wire, Ben Shapiro, Glenn Beck, Jesse Lee Peterson, Breitbart News, The Western Journal, The Daily Signal"]]

## data for the hover
customdata = np.dstack((z2, z3))

## Creating the heatmap
fig = px.imshow(a,labels=dict(x="Political Bias", y="Community", color="Number of channels"),
                text_auto=True,origin='lower',color_continuous_scale='Blues');

## Creating the figure
fig=fig.add_trace(go.Heatmap(
    z=a,
    customdata=np.dstack((z2, z3)),
    hovertemplate='%{customdata[0]}</b>',
    coloraxis="coloraxis1", name='a',
    text=a,
    texttemplate="%{text}")
    );

## Add the title
fig=fig.update_layout(title_text='Political orrientation of the comunities');

## Display
fig.show()


# %%
## Displaying the number of subsribers per community

df=news_channels_com.groupby(by='community').sum()
df=df.drop('channel_num',axis=1)


df['community']=range(0,6)

fig = px.bar(df,x='community',y='subscribers_cc',
            color='community', color_continuous_scale='balance_r',title='Number of subscribers per community',
            labels={
                     "community": "Community",
                     "subscribers_cc": "Number of subscribers"}
             )

fig=fig.update(layout_coloraxis_showscale=False)
fig=fig.update_xaxes(tickvals=np.arange(6))

#fig.write_html('subscribers_community.html')

fig.show()
# %%

## Displaying the number of videos per community

df=news_channels_com.groupby(by='community').sum()
df['community']=range(0,6)
fig = px.bar(df,x='community',y='videos_cc',
            color='community', color_continuous_scale='balance_r',title='Number of videos per community',
            labels={
                     "community": "Community",
                     "y": "Number of videos per community"}
             )
fig=fig.update(layout_coloraxis_showscale=False)
fig=fig.update_xaxes(tickvals=np.arange(6))
#fig.write_html('channel_community.html')
fig.show()

# %%
## Display to see what we are working on
news_channels_com.sort_values(by=['community','subscribers_cc'],ascending=False)

## Import Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image


## Import the mask of the wordcloud
mask = np.array(Image.open('../figures/usa_map.jpg'))

## Crete function for gettin a cloud of words
def create_wordcloud(df,community,color_map):
    ## Select the community
    fil=df.loc[df['community']==community]

    ## Set weight 
    mydict=fil.set_index('name_cc').to_dict()['subscribers_cc']

    ## Creating the cloud of words
    wc = WordCloud( width=800,
                    height=800,
                    max_words=100,
                    colormap=color_map,
                    background_color=None,
                    mode="RGBA",mask=mask).generate_from_frequencies(mydict)

    fig=plt.imshow(wc)

    ## Layout 
    fig=plt.axis("off")
    fig=plt.title("Community {}".format(community))

    ## Saving
    #plt.savefig("pictures/wordclouds_com_{}".format(community),bbox_inches='tight')
    
    return fig

# %%
## Creating a subplot with the 6 communities

## Seting colors
colors=["GnBu","Purples","Greens","YlOrBr","Oranges","Reds"]

## Create subplot
fig, ax = plt.subplots(3,1, figsize=(10,12));

## Set transparency of the backgroud
fig.patch.set_alpha(0);

## Loop to apply the create_wordcloud function to each community
for i in range(0,6):
    plt.subplot(3, 2, i+1);
    ax=create_wordcloud(news_channels_com,i,colors[i]);

## Saving
#fig.savefig("../ste_website/ADAmantiumForgers/assets/img/wordclouds_merged_transparent.png",bbox_inches='tight',dpi=1200)
# %%
