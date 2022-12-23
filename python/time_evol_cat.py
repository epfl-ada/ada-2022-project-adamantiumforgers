# %%
'''
File name: time_evol_cat.py
Author: Loïc Fischer
Date created:16/12/2022
Date last modified: 20/12/2022
Python Version: 3.9.13

The purpose of this file is to give an overview
of the evolution of the channel categories throught time.

A run version is presented in time_evol_cat.ipynb
'''

# %%
## Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
import sys
import json
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

## Read data from memory and clean it
df_channels = pd.read_csv(os.path.join(PATH,DATA_CHANNEL), compression=COMPRESSION,sep=SEPARATOR)
df_channels = df_channels.dropna()

## Read data from memory and clean it only a part is imported du to memory issues
## The whole data imported later using chunks
df_video_meta = pd.read_json(os.path.join(PATH,DATA_VIDEO_META), compression=COMPRESSION,lines=True, nrows=10000)
df_video_meta = df_video_meta.dropna()

## Read data from memory and clean it only a part is imported du to memory issues
## The whole data imported later using chunks
df_comments = pd.read_csv(os.path.join(PATH,DATA_COMMENT),sep=SEPARATOR,nrows=1e6)
df_comments = df_comments.dropna()

## Display head of DataFrame

df_channels.head(2)
df_video_meta.head(2)
df_comments.head(2)

# %%
## Creating a empty dataset

channels_cat=list(df_channels.category_cc.unique())

## Renaming the caterogy to fit the category in video metadata
channels_cat.remove('Film and Animation')
channels_cat.append('Film & Animation')

## Setting years
years=list(range(2005,2020))

## Creating an empty data in the wanted shape
cross_tab=pd.DataFrame(np.zeros((len(years),len(channels_cat))),years,channels_cat)
cross_tab

# %%
## iterating over the  video metadata
## Commented because of the time to run: aroud 40 minutes
'''
## Initialize i
i=0

## Import the empty dataframe
cross_tab_it=cross_tab

## Create chunks to read the data
chunks = pd.read_json(os.path.join(PATH,DATA_VIDEO_META),compression=COMPRESSION, lines=True, chunksize = 1e5)

## loop on chung to read the data and create a crosstab
for chunk in chunks:

    ## Getting values from the previous loop
    cross_tab_old=cross_tab_it

    ## droping useless data
    chunk=chunk.drop(['channel_id','crawl_date','description','dislike_count','duration','title','like_count','tags'],axis=1)

    ##Changing type to datetime
    chunk['upload_date']=pd.to_datetime(chunk['upload_date'])

    ## Creating crosstab
    cross_tab_new=pd.crosstab(chunk.upload_date.dt.year,chunk.categories)
    a=chunk.groupby(by=[chunk.upload_date.dt.year,chunk.categories])
    a.size()
    cross_tab_new
    cross_tab_it=cross_tab_old.add(cross_tab_new,fill_value=0)

    ## Print i to have the progress report
    i=i+1
    print(i)

    ## break for the trials
    if i >9 :
        break

print(view_count)

display(cross_tab_it)

## Saving to csv
#cross_tab_it.to_csv(csv_outputs/cat_evolution.csv)
'''


# %%
#importing and filtering data from csv

## Initilize of dataframe
evolution=pd.DataFrame

## Read the data from the previous cell
evolution=pd.read_csv('../csv_outputs/evolution.csv',index_col=0)

## Fill NaN with 0
evolution=evolution.fillna(0)

#Sorting on values from 2018
evolution=evolution.sort_values(by = 2018, axis = 1,ascending = False)

## Drop 2019 since the year is not complete
evolution=evolution.drop(2019,axis=0)

## Display DataFrame
evolution

# %%
## Static line plot


plt.figure();
sns.set_style("whitegrid")
ax=sns.lineplot(data=evolution)

## Layout
ax=sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
ax=plt.ylabel('Number of videos uploaded');
ax=plt.xlabel('Year');
ax=plt.title('Evolution of categories');


## Saving figure
#ax=plt.savefig('pictures/category_evolution.png',bbox_inches='tight',dpi=150)

## Display it
plt.show()
# %%
## Interactive line plot

fig = px.line(evolution,title='Evolution of Youtube categories',
            labels={"index": "Year",
                    "value": "Number of videos uploaded",
                    "variable":"Category"
                    }
            )
fig.show()

## Saving figure:
#fig.write_html("html_figures/overall_categories_evolution.html")

# %%
## Iterrating video metadata over a particular year

## This basically the same code as befor but with a limitation on the year

'''
## format: "mm.dd.yyyy"
start_date=pd.to_datetime("01.01.2019")
end_date=pd.to_datetime("10.01.2019")

months=list(range(1,13))
cross_tab_it=pd.DataFrame(np.zeros((len(months),len(channels_cat))),months,channels_cat)

i=0
appended_videos=pd.DataFrame()
chunks = pd.read_json(os.path.join(PATH,DATA_VIDEO_META),compression=COMPRESSION, lines=True, chunksize = 1e5)

for chunk in chunks:
    cross_tab_old=cross_tab_it
    chunk=chunk.drop(['channel_id','description','dislike_count','duration','title','like_count','tags'],axis=1)
    chunk['upload_date']=pd.to_datetime(chunk['upload_date'])
    chunk_filtered=chunk[(chunk['upload_date']>=start_date)&(chunk['upload_date']<end_date)]
    cross_tab_new=pd.crosstab(chunk_filtered.upload_date.dt.month,chunk_filtered.categories)
    cross_tab_it=cross_tab_old.add(cross_tab_new,fill_value=0)
    i+=1
    #if i>20:
    #    break
    print(i)

#cross_tab_it.to_csv('csv_outputs/{}_month_category_evo.csv'.format(start_date.year))

'''

# %%
##  Static plot of evolution over 2019
for i in [2018,2019]:
    selected_year=i

    ## Read data and clean it
    cross_tab_it=pd.read_csv('../csv_outputs/{}_month_category_evo.csv'.format(selected_year),index_col=0)
    cross_tab_it=cross_tab_it.drop('Unnamed: 1',axis=1)

    evolution_201X=cross_tab_it

    ## Condition on year to have a better display
    if selected_year == 2019:
        evolution_201X=evolution_201X.sort_values(by = 9, axis = 1,ascending = False)
        evolution_201X=evolution_201X.drop([10,11,12],axis=0)
    if selected_year == 2018:
        evolution_201X=evolution_201X.sort_values(by = 12, axis = 1,ascending = False)
    
    ## Creating the plot
    sns.set_style("whitegrid")
    plt.figure();
    ax=sns.lineplot(data=evolution_201X)

    ## Layout
    ax=sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
    ax=plt.ylabel('Number of videos uploaded');
    ax=plt.xlabel('month');
    ax=plt.title('Evolution of categories in {}'.format(i));
    #ax=plt.savefig('pictures/{}_category_evolution.png'.format(selected_year),bbox_inches='tight',dpi=150)

    ## Display
    plt.show()
# %%
# %%
## Interactive Evolution over 2018 and 2019

for i in [2018,2019]:
    selected_year=i

    ## Read data and clean it
    cross_tab_it=pd.read_csv('../csv_outputs/{}_month_category_evo.csv'.format(selected_year),index_col=0)
    cross_tab_it=cross_tab_it.drop('Unnamed: 1',axis=1)
    evolution_201X=cross_tab_it
    
    ## Condition on year to have a better display
    if selected_year == 2019:
        evolution_201X=evolution_201X.drop([10,11,12],axis=0)
        evolution_201X=evolution_201X.sort_values(by = 1, axis = 1,ascending = False)
    
    if selected_year == 2018:
        evolution_201X=evolution_201X.sort_values(by = 12, axis = 1,ascending = False)

    ## Creating the plot
    fig = px.line(evolution_201X,title='Evolution of categories in {}'.format(selected_year),
                labels={"index": "Month",
                        "value": "Number of videos uploaded",
                        "variable":"Category"
                        }
                );

    ## Layout
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [1,2,3,4,5,6,7,8,9,10,11,12],
            ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        ));
    plt.show()

    #Save figure
    #fig.write_html("html_figures/{}_category_evolution.html".format(selected_year));
# %%
