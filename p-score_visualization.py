
# %% 1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% 2

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

## Filter
min_comments = 10
max_comments = 10000
author_pscore = author_pscore[author_pscore['num_comments'] > min_comments]
author_pscore = author_pscore[author_pscore['num_comments'] < max_comments]
author_pscore = author_pscore[author_pscore['year'] >= 2015]


display(author_pscore.head(10))
print(f'Number of authors: {len(author_pscore)}')

plt.scatter(author_pscore['p_score'],author_pscore['num_comments'],s=1)
plt.xlabel('p-score')
plt.ylabel('Number of comments')
plt.show()

#%%
# years = np.arange(2015,2020) # 2005
# display(years)

# for year in years:
#     author_pscore_year = author_pscore[author_pscore['year'] == year]
#     author_pscore_year.hist('p_score', bins=50)
#     plt.title(year)

#%%
import plotly.express as px
#df = px.data.gapminder()

#author_pscore = author_pscore[author_pscore['year'] == '2016']
author_pscore = author_pscore.sort_values(by='year', ascending=False)

fig = px.histogram(author_pscore,
                   x="p_score",
                   nbins=100,
                   title='Histogram of polarization scores',
                   animation_frame="year",
                   range_x=[-1,1],
                   opacity=0.8,
                   marginal="box")
                   #histnorm='percent')

yranges = {2019: [0, 8548],
           2018: [0, 9509],
           2017: [0, 2858],
           2016: [0, 1671],
           2015: [0, 778]}

# for f in fig.frames:
#     if int(f.name) in yranges.keys():
#         f.layout.update(yaxis_range = yranges[int(f.name)])

# fig.update_layout(
#     autosize=False,
#     width=750,
#     height=500
#     )

fig.show()

#%%
import plotly.io
#plotly.io.write_html(fig, 'p-score.html', auto_play=False)

## Do better plots:
## Number of comments follows a power law

# %%

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
df = px.data.tips()

fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.5, 1])

fig1 = px.ecdf(df, x="total_bill", color="sex", markers=True, lines=False, marginal="histogram")
fig2 = px.ecdf(df, x="total_bill", color="sex", markers=True, lines=False, marginal="rug")

## rug plot components
fig.add_trace(fig2.data[1],row=1, col=1)
fig.add_trace(fig2.data[3],row=1, col=1)

## histogram components 
fig.add_trace(fig1.data[1], row=2, col=1)
fig.add_trace(fig1.data[3], row=2, col=1)

## cumulative distribution scatter
fig.add_trace(fig1.data[0], row=3, col=1)
fig.add_trace(fig1.data[2], row=3, col=1)

fig['layout']['barmode'] = 'overlay'
fig['layout']['xaxis3']['title'] = 'total_bill'
fig['layout']['yaxis3']['title'] = 'probability'
fig.show()

# %%

import seaborn as sns

# Load one of the data sets that come with seaborn
tips = sns.load_dataset("tips")

sns.jointplot("p_score", "num_comments", author_pscore, kind='reg')

# %%
#sns.kdeplot(x=author_pscore['p_score'], y=author_pscore['num_comments'])
#sns.histplot(x=author_pscore['p_score'], y=author_pscore['num_comments'], bins=100)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, nrows=1, figsize=(15, 6), sharex=True, sharey=True)
plt.title('Polarization score')
plt.xlabel('p-score')
plt.ylabel('Author count')

sns.histplot(author_pscore[author_pscore['year'] == 2019]['p_score'], stat='probability', binrange=[-1,1] , kde=True, ax=ax1, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2018]['p_score'], stat='probability', kde=True, ax=ax2, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2017]['p_score'], stat='probability', kde=True, ax=ax3, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2016]['p_score'], stat='probability', kde=True, ax=ax4, bins=30)
sns.histplot(author_pscore[author_pscore['year'] == 2015]['p_score'], stat='probability', kde=True, ax=ax5, bins=30)



#plt.yscale('log')

#%%
author_pscore2 = author_pscore[author_pscore['year'] > 2014]

sns.histplot(author_pscore2,
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
import plotly.figure_factory as ff

x = np.random.randn(1000)
x = np.array(author_pscore['p_score'])
hist_data = [x]
group_labels = ['distplot'] # name of the dataset

fig = ff.create_distplot([x], group_labels, bin_size=.1)
fig.show()


#%%
fig = go.Figure()
fig.add_trace(go.Histogram(x = author_pscore['p_score'], 
                           histnorm='percent'))
fig.update_layout(xaxis_title="p-score", yaxis_title="Users")
fig.show()
# %%
