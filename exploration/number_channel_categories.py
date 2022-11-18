import numpy as np
import pandas as pd
from tqdm import tqdm

DIR_LARGE = "data/large/"

PATH_METADATA = DIR_LARGE + "yt_metadata_en.jsonl.gz"

uniques = set()
for chunk in tqdm(pd.read_json(PATH_METADATA, compression='infer', lines=True, chunksize=10000)) :
    uniques.update(chunk.categories.unique())
uniques