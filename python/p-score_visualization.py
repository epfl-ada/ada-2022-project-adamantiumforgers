
# %%
## Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



# %% 
## Directories for the data.
DIR = "data/"
DIR_OUTPUTS = "csv_outputs/"

## Path for each file
PATH_PSCORE = DIR_OUTPUTS + "author_pscore.csv"
PATH_PSCORE_FULL = DIR_OUTPUTS + "author_pscore_full.csv"


# %%
## Data visualization
## Read the p_score file
author_pscore = pd.read_csv(PATH_PSCORE_FULL, sep=";")

## Filter the number of comments and year
min_comments = 25
max_comments = 10000
author_pscore = author_pscore[author_pscore['num_comments'] > min_comments]
author_pscore = author_pscore[author_pscore['num_comments'] < max_comments]
author_pscore = author_pscore[author_pscore['year'] >= 2015]

## Overview of the data
display(author_pscore.head(10))
print(f'Number of authors: {len(author_pscore)}')


#%%
## Plot the distribution of the number of comments per author
plt.scatter(author_pscore['p_score'],author_pscore['num_comments'],s=1)
plt.title('Authors placed as a function of their polarization and activity')
plt.xlabel('p-score')
plt.ylabel('Number of comments')
plt.show()


#%%
## Plot the p-scores distribution for each year
years = np.arange(2015,2020) # 2005
display(years)

for year in years:
    author_pscore_year = author_pscore[author_pscore['year'] == year]
    author_pscore_year.hist('p_score', bins=50)
    plt.title(year)


#%%
## Interactive plots
## 
import plotly.express as px

#author_pscore = author_pscore[author_pscore['year'] == '2016']
author_pscore = author_pscore.sort_values(by='year', ascending=True)

fig = px.histogram(author_pscore,
                   x="p_score",
                   nbins=100, #max number of bins (automatic)
                   title='Histogram of polarization scores',
                   animation_frame="year",
                   range_x=[-1,1],
                   opacity=0.8,
                   marginal="box")
                   #histnorm='percent')

## Adjust the scale for each year
## Prefered instead of normalizing to keep the information about the real number of the bin size
yranges = {2015: [0, 778],
           2016: [0, 1671],
           2017: [0, 2858],
           2018: [0, 9509],
           2019: [0, 8548]
           }

for f in fig.frames:
    if int(f.name) in yranges.keys():
        f.layout.update(yaxis_range = yranges[int(f.name)])

## Add text and different layout
fig.update_layout(autosize=False,
                  width=750,
                  height=700,
                  title={'text' : 'Histogram of polarization scores',
                         'x':0.5,
                         'xanchor': 'center'
                         },
                  xaxis_title="p-score",
                  yaxis_title="Number of users")
                  #sliders=[dict(active=4)])

fig["layout"].pop("updatemenus")
fig.show()


#%%
## Export the figure in html format
plotly.io.write_html(fig, 'p-score.html', auto_play=False)


# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(15, 6), sharex=True, sharey=True)
plt.title('Polarization score')
plt.xlabel('p-score')
plt.ylabel('Author count')

sns.histplot(author_pscore[author_pscore['year'] == 2019]['p_score'], stat='probability', binrange=[-1,1] , kde=True, ax=ax1, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2018]['p_score'], stat='probability', kde=True, ax=ax2, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2017]['p_score'], stat='probability', kde=True, ax=ax3, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2016]['p_score'], stat='probability', kde=True, ax=ax4, bins=30)


#%%
sns.histplot(author_pscore,
             x='p_score',
             hue='year',
             palette='rainbow',
             #multiple="layer",
             #kde=True, 
             #element="poly",
             bins=40,
             stat="percent",
             #fill=False,
             #thresh=2,
             common_norm=False)
#sns.displot(penguins, x="flipper_length_mm", hue="species", element="step")
# %%
