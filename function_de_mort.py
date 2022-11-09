# %%
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
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

PATH_METADATA_HELPER = DIR + "yt_metadata_helper.feather"
PATH_COMMENTS = DIR_LARGE + "youtube_comments.tsv.gz"
PATH_METADATA = DIR_LARGE + "yt_metadata_en.jsonl.gz"
PATH_RAW_METADATA = DIR_LARGE + "raw_yt_metadata.jsonl.zst"

PATH_AUTHOR_TO_CHANNEL = DIR + "authors_to_channel.csv"



# %%
#for comments in pd.read_csv(PATH_COMMENTS, compression="infer", sep="\t", chunksize=10000):

# %%
df_metadata = pd.read_json(PATH_METADATA, compression='infer', lines=True, nrows=10000)
df_metadata.head()

# %%
df_author_video = pd.read_csv(PATH_AUTHOR_TO_CHANNEL, compression="infer", sep="\t", nrows=10000)
df_author_video.head()

# %%
author_video = pd.DataFrame(columns=['author','channel_id'])
author_video.to_csv("data/author_to_channel_id.csv",sep=";",header=True, index=False)
channel_num = pd.read_csv(DIR + "channels.csv")
print(channel_num.index)
print(df_author_video.index)
print(df_author_video.columns)

# %%
print
for chunk in pd.read_csv(PATH_AUTHOR_TO_CHANNEL, compression="infer", sep="\t", chunksize=10000):
    to_save = chunk.groupby(channel_num, on='channel_num')
    to_save.to_csv(DIR + 'author_to_channel_id.csv', mode='a')


# %%
