
# %% 1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %% 2
## Directories for the data.
## Large data can be stored on an external drive and accessed by creating a simlink to a "large" directory in the data folder
## ln -s target_path link_path

DIR = "data/"
DIR_OUTPUTS = "csv_outputs/"
DIR_LARGE = "data/large/"

## Path for each file
PATH_TIME_SERIES = DIR + "df_timeseries_en.tsv.gz"
PATH_CHANNELS = DIR + "df_channels_en.tsv.gz"
PATH_NUM_COMMENTS = DIR + "num_comments.tsv.gz"
PATH_NUM_COMMENTS_AUTHORS = DIR + "num_comments_authors.tsv.gz"
PATH_METADATA = DIR_LARGE + "yt_metadata_en.jsonl.gz"
PATH_METADATA_HELPER = DIR + "yt_metadata_helper.feather"
PATH_COMMENTS = DIR_LARGE + "youtube_comments.tsv.gz"

PATH_AUTHORS = DIR_OUTPUTS + "authors_to_channels.csv"
PATH_MEDIAS_VIDEOS = DIR_OUTPUTS + "display_id_to_medias.csv"
PATH_MEDIAS_VIDEOS_YEAR = DIR_OUTPUTS + "display_id_to_medias_year.csv"
PATH_PSCORE = DIR_OUTPUTS + "author_pscore.csv"
PATH_PSCORE_FULL = DIR_OUTPUTS + "author_pscore_full.csv"


# %% 3
display_id_to_medias = pd.read_csv(PATH_MEDIAS_VIDEOS_YEAR, sep=";")
allsides_medias = pd.read_csv("allsides_scraping/csv/channels_w_pol_orr.csv", sep=",")

## Polarization score function
def p_rating(orientation):
    # match orientation: ##only works with the latest python version
    #     case 'left': return -1
    #     case 'lean left': return -0.5
    #     case 'center': retunr 0
    #     case 'lean right': return 0.5
    #     case 'right': return 1
    #     case _: return 0
    if orientation == 'Left': return -1
    elif orientation == 'Lean Left': return -0.5
    elif orientation == 'Center': return 0
    elif orientation == 'Lean Right': return 0.5
    elif orientation == 'Right': return 1

## Create column with orientation value
allsides_medias['orientation_num'] = allsides_medias['orrientation'].apply(p_rating)
display(allsides_medias.head())
display(display_id_to_medias.head())


# %% 4
# Erase previous versions
pd.DataFrame(columns=['author','year','score','num_comments']).to_csv(PATH_PSCORE,sep=";",header=True, index=False)

i=1
## On average, it finds 1 comment on a video from Allsides over 1000 comments
for chunk in pd.read_csv(PATH_COMMENTS, sep='\t', usecols = ['author','video_id'], chunksize=3e6):
    
    ## Keep track of the loop
    print(f'Iteration: {i}')
    i=i+1
    ## nrows=1e8: 34 iterations
    ## nrows=1e9: 334 iterations
    ## nolimit: 2869 iterations

    ## Merge to retrieve the channel_id from the video_id (only if the video is from a media)
    df_temp = chunk.merge(display_id_to_medias, left_on='video_id', right_on='display_id')
    ## Merge to retrieve the political orientation (only if the video is from a AllSides channel)
    df_temp = df_temp.merge(allsides_medias[['channel', 'orientation_num']], left_on='channel_id', right_on='channel')

    ## Keep only relevant columns: author, orientation_num and year
    df_temp = df_temp[['author','orientation_num','upload_date']]
    df_temp['num_comments'] = 1
    df_temp = df_temp.rename(columns={'orientation_num': 'score', 'upload_date': 'year'})
    recall = df_temp

    ## Group by author: 1/ p-score = sum of orientation_num 2/ Count the number of comments
    ## as_index = False is used to flaten the column names
    #df_temp = df_temp.groupby('author', as_index=False).agg({'score':'sum', 'num_comments':'count'})  ## To use if the distinction by years is irrelevant
    df_temp = df_temp.groupby(['author','year'], as_index=False).agg({'score':'sum', 'num_comments':'count'})

    #recall = df_temp.sort_values(by='author', ascending=True)

    ## Save to csv
    df_temp.to_csv(PATH_PSCORE,sep=";",mode='a',header=False, index=False)

## See what happens (keep an eye on the data structure and the preliminary results)
display(recall.head(10))

#%%
## Postprocess
## Read the p_score file
author_pscore = pd.read_csv(PATH_PSCORE_FULL, sep=";")
## Group_by author and year
author_pscore = author_pscore.groupby(['author','year'], as_index=False).agg({'score':'sum', 'num_comments':'sum'})

## Compute the pscore and sort by highest values
author_pscore['p_score'] = author_pscore['score']/author_pscore['num_comments']
author_pscore = author_pscore.sort_values(by='p_score', ascending=True)

## Save
author_pscore.to_csv(PATH_PSCORE_FULL,sep=";",header=True, index=False)