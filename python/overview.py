# %%
'''
File name: news_channel.py
Author: Loïc Fischer
Date created: 05/11/2022
Date last modified: 22/12/2022
Python Version: 3.9.13

A fully compiled an commented version is available: overview.ipynb
'''

# %%
## Import Libraries

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc

import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

import os

## Show more than one display per cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% 
## Import the data

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


## Display the first rows of the dataframes
df_channels.head(2)
df_video_meta.head(2)
df_comments.head(2)

# %%
## Pie chart number of channels (not used finnaly)

""" 
palette_color =sns.color_palette('tab20')
g=df_channels.groupby(by=['category_cc'])
a=g.count().sort_values(by='subscribers_cc',ascending=False)
my_labels=list(a['videos_cc'].index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(a['subscribers_cc'],labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);
 """
# %%
## Pie chart number of subscribers (not used finnaly)


""" 
palette_color =sns.color_palette('tab20')
g=df_channels.groupby(by=['category_cc'])
a=g.sum().sort_values(by='subscribers_cc',ascending=False)
my_labels=list(a['subscribers_cc'].index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4

plt.pie(a['subscribers_cc'],labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);
"""

# %%
## Pie chart number of videos (not used finnaly)

""" 
palette_color =sns.color_palette('tab20')
g=df_channels.groupby(by=['category_cc'])
a=g.sum().sort_values(by='videos_cc',ascending=False)
my_labels=list(a['videos_cc'].index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(a['videos_cc'],labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);
 """

# %%
## Count of the total number of views per category
## Has been commented due to it long time to run

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
#view_count.to_csv('data/view_count.csv')
'''

# %%
## Total number of views (not used finnaly)

""" 
view_cat=pd.read_csv('data/view_count.csv')
view_cat=view_cat.dropna()
view_cat=view_cat.sort_values(by='view_count',ascending=False)
my_labels=list(view_cat.categories)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
plt.pie(view_cat.view_count,labels=my_labels,rotatelabels=True,explode=my_explode,colors=palette_color,startangle=90,counterclock=False);
 """

# %%
##Import data

view_cat=pd.read_csv('../csv_outputs/view_count.csv')
view_cat=view_cat.dropna()
view_cat=view_cat.sort_values(by='view_count',ascending=False)


# %%
## Interactive Pie Chart with the total number of videos.

a=df_channels.groupby(by=['category_cc']).sum().sort_values(by='videos_cc',ascending=False)
my_labels=list(a.index)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
pie_videos=go.Pie(   labels=my_labels,
                                values=a['videos_cc'],
                                marker_colors=px.colors.qualitative.G10,
                                pull=my_explode,
                                title='Number of videos'
                                )
fig_video = go.Figure(data=[pie_videos])
fig_video.update_layout(
    autosize=False,
    width=750,
    height=500
    )
#fig_video.write_html('../html_figures/pie_videos.html')
fig_video.show();
# %%
## Interactive Pie Chart with the total number of views.

my_labels=list(view_cat.categories)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
pie_view=go.Pie(   labels=my_labels,
                                values=view_cat.view_count,
                                marker_colors=px.colors.qualitative.G10,
                                pull=my_explode,
                                title='Number of views'
                                )

fig_view = go.Figure(data=[pie_view])
fig_view.update_layout(
    autosize=False,
    width=750,
    height=500
    );
#fig_view.write_html('../html_figures/pie_views.html');
fig_view.show();

# %%
## Interactive Pie Chart with the total number of channels.

a=df_channels.groupby(by=['category_cc']).count().sort_values(by='subscribers_cc',ascending=False)
my_labels=list(view_cat.categories)
my_explode=np.zeros(len(my_labels))
my_explode[my_labels.index('News & Politics')]=0.4
pie_channels=go.Pie(   labels=my_labels,
                                values=a.channel,
                                marker_colors=px.colors.qualitative.G10,
                                pull=my_explode,
                                title='Number of channels'
                                )
fig_channels = go.Figure(data=[pie_channels])
fig_channels.update_layout(
    autosize=False,
    width=750,
    height=500
    );
#fig_channels.write_html('../html_figures/pie_channels.html');
fig_channels.show();

# %%
## Creating a subplot with the pie charts

fig = make_subplots(rows=3, cols=1,specs=[[{"type": "pie"}], [{"type": "pie"}],[{"type": "pie"}]])

fig.add_trace(pie_videos, row=1, col=1);

fig.add_trace(pie_view, row=2, col=1);

fig.add_trace(pie_channels, row=3, col=1);


fig.update_layout( title_text="Overview of News & Politics category",height=1400)#, width=1500,)
fig.write_html('../html_figures/news_pol_3pie_supperposed.html')
fig.show();

# %%
data = {
'Age':['18-29','30-49','50-64','65+'],
'TV': [27,45,72,85],
'Online': [50, 49, 29,20],
'Radio': [14,27, 29, 24],
'Print newspapers': [5, 10, 23, 48]
}

news_sources=pd.DataFrame(data)
x=news_sources.columns

# %%
fig = px.bar(news_sources, x='Age', y=['TV','Online','Radio','Print newspapers'] ,  title="News media per age")
fig.show()

# %%
""" 
## Data
r = [0,1,2,3]

## plot

barWidth = 0.85
names = ['18-29','30-49','50-64','65+']
## Create Online Bars
plt.bar(r, Online, edgecolor='white', width=barWidth, label='Online')
## Create TV Bars
plt.bar(r, TV, bottom=Online, edgecolor='white', width=barWidth, label='TV')
## Create Radio Bars
plt.bar(r, Radio, bottom=[i+j for i,j in zip(TV, Online)], edgecolor='white', width=barWidth, label='Radio')
## Create Newspapers Bars
plt.bar(r, Print_newspapers, bottom=[i+j+k for i,j,k in zip(TV, Online,Radio)], edgecolor='white', width=barWidth, label='Printed newspaper')
## Custom x axis
plt.xticks(r, names)
plt.title('Main source of news per age')

## Add a legend
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
sns.set_palette("bwr",n_colors=4) 
## Save plot
plt.savefig('pictures/main_source_news_age.png')

## Show graphic
plt.show();
 """
# %%


# pandas dataframe
sns.set_palette("bwr",n_colors=4) 

raw_data = {    'TV': [27,45,72,85],
                'Online': [50, 49, 29,20],
                'Radio': [14,27, 29, 24],
                'Print newspapers': [5, 10, 23, 48]}

df = pd.DataFrame(raw_data)
 
## From raw value to percentage
totals = [i+j+k+l for i,j,k,l in zip(df['TV'], df['Online'], df['Radio'],df['Print newspapers'])]
TV = [i / j * 100 for i,j in zip(df['TV'], totals)]
Online = [i / j * 100 for i,j in zip(df['Online'], totals)]
Radio = [i / j * 100 for i,j in zip(df['Radio'], totals)]
Print_newspapers= [i / j * 100 for i,j in zip(df['Print newspapers'], totals)]
 
## Creating dataframe
df = pd.DataFrame(data={'Online':Online, 'TV': TV, 'Radio': Radio, 'Newspaper': Print_newspapers})
df.index = ['18-29','30-49','50-64','65+']

## Creating the bars
ax = df.plot(kind='bar', stacked=True, figsize=(8, 6), rot=0, xlabel='Age group', ylabel='%')
for c in ax.containers:

    ## Optional: if the segment is small or 0, customize the labels
    labels = [round(v.get_height(),1) if v.get_height() > 0 else '' for v in c]
    
    ## remove the labels parameter if it's not needed for customized labels
    ax.bar_label(c, labels=labels, label_type='center')

## Layout
plt.title('Main source of news per age')
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


## Save plot
plt.savefig('pictures/main_source_news_age.png',bbox_inches='tight')

plt.show();
# %%
## Initialize data for the bar plot
## Source: https://www.pewresearch.org/journalism/2021/01/12/news-use-across-social-media-platforms-in-2020/pj_2021-01-12_news-social-media_0-02/

data = {
'social_media':['Facebook','Youtube','Twitter','Instagram','Reddit','Snapchat','LinkedIn','TikTok','WhastApp'],
'visit': [68,74,25,40,15,22,25,12,19],
'news': [36, 23, 15,11,6,4,4,3,3],
}
social_media_as_news=pd.DataFrame(data)
# %%

##Bar plot

## set plot style: grey grid in the background:
sns.set(style="darkgrid")

## set the figure size
plt.figure(figsize=(14, 14))


## bar chart 1 -> top bars
bar1 = sns.barplot(x="social_media",  y="visit", data=social_media_as_news, color='darkblue')


## bar chart 2 -> bottom bars
bar2 = sns.barplot(x="social_media", y="news", data=social_media_as_news, estimator=sum, ci=None,  color='lightblue')

## add legend
top_bar = mpatches.Patch(color='darkblue', label='Use site')
bottom_bar = mpatches.Patch(color='lightblue', label='Regulary get news on site')
legend=plt.legend(handles=[top_bar, bottom_bar],title ="% of Adults who :\n",fontsize=15)
plt.setp(legend.get_title(),fontsize=15);

## Layout of plot
bar1.set(xlabel=None,ylabel=None)

## Add a titlte
plt.title('Sources of news on social media',fontsize = 20)

## Save figure
plt.savefig('../../ste_website/ADAmantiumForgers/assets/img/sources_of_news.png', bbox_inches='tight')

## show the graph
plt.show();


# %%
