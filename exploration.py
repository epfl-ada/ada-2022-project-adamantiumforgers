# %% [markdown]
# # Michel exploration
# 
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
df_raw_metadata = pd.read_json(PATH_RAW_METADATA, compression='infer', lines=True, nrows=10000)

# %%
df_channels.head()
# %%
df_timeseries.head()
# %%
df_comments.head()
# %%
df_num_comments.head()
# %%
df_num_comments_authors.head()
# %%
df_metadata.head()
# %%
df_comments[df_comments['author']==2].head()
# %%
