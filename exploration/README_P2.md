# Does Youtube reflect the overall political polarization in the US?

By the Adamantium Forgers:

| Name                | Email                       |
|---------------------|-----------------------------|
| Loïc Fischer        | loic.fischer@epfl.ch        |
| Stéphane Weissbaum  | stephane.weissbaum@epfl.ch  |
| Camille Bernelin    | camille.bernelin@epfl.ch    |
| Michel Morales      | michel.morales@epfl.ch      |


## Abstract
In 2017, with the arrival in power of Donald Trump as president of the United States, the American political world quickly split between the pro and anti-Trump. According to a [study][1] lead by Jesse M. Shapiro, Brown University, this polarization began in the late 1990s and early 2000s and has been only increasing since, promoted by the detrimental properties of the US voting system that incentivize people to become radical. This polarization is also reflected online: according to a [study][2] on Facebook lead by to Brazilian researcher, the polarization one year after the 2017 election can be pictured on the figure below. Can the same effect be measured on YouTube communities ? In this data story, we will analyze the profiles of political communities on YouTube.


<img src="./pictures/fb_us_pol.png" alt="fb_us_pol" width="700"/>

*Complete disconnexion of political landscape. Pablo Ortellado and Marcio Moretto Ribeiro, CC BY*


## Research Questions
We will try to answer the following questions, in accordance to the time available before the milestone 3:
1. Can we identify communities inside the main US News&Politics channels ? Is there a clear left-right polarization or is the polarization independent of classical political party confrontation ?
2. What is the distribution of users polarization? Is it Gaussian? How many very extreme users do we see? For this purpose, we will assign a p-score (polarization or political score) to users.
3. Based on the communities identified, can we describe political orientation of audience of non-News&Politics channels ? Study a few particular «neutral» channels (ex : Education, Gaming, Comedy Channels ...) or marginal channels (Flat Earth Society) and analyze their audience and their links with news channels.
4. How are main political events treated by channels in different communities ? Study various events by filtering by tags, descriptions or video titles that contains key words, for example "US Election", "Impeachment", "Hunter Biden", "Capitol".


## Additional dataset 

In order to classify the different political orientations of the youtube users, we decided to use the media bias classification given by [Allsides][3]. AllSides is an American company that assesses the political bias of prominent media outlets, categorized in five political orientation categories: Left, Lean Left, Center, Lean Right, Right. More info [here][5]. The code used to extract the AllSides dataset is in the [allsides_scraping.py](./allsides_scraping/allsides_scraping.py) file. No notebook was produced since no output is required, only an export to csv.

<img src="./pictures/media_bias_allsides.png" alt="media_bias" width="400"/>

*US media bias classification* 

The website has been scraped and 2 data sets have been created:
- A dataset with only the featured media (59 medias)
- A dataset with all the media identified on the website (1279 medias)

5 kind of data have beeen extracted from the website regarding the medias:
- name
- political bias given by [Allsides][3]
- confidence level on bias of [Allsides][3]
- number of community votes confirming the media bias
- number of community votes disproving the media bias


### Data collected from Allsides:
|        |**name**|**orrientation**|**confidence**|**commu_agree**|**commu_disagree**|
|--------|------------------------|------------|----------|---------|------------|
| **0**  | ABC News               | Lean Left  | High     | 39305   | 19628      |
| **...**| ...                    | ...        | ...      | ...     | ...        |


Once the dataset cleaned, 37 on 59 medias have associated channels in the youniverse dataset. Those csv files, [*channels_yt_all.csv*](./allsides_scraping/csv/channels_yt_all.csv) and [*channels_yt_ft.csv*](./allsides_scraping/csv/channels_yt_ft.csv), can be found [here](./allsides_scraping/csv).

### Merged data with the Youniverse dataset:
||**category_cc**|**join_date**|**channel**|**name_cc**|**subscribers_cc**|**videos_cc**|**subscriber_rank_sb**|**weights**|**orrientation**|**confidence**|**comu_agree**|**comu_disagree**|
|--|---|---|----|-----|----|----|----|----|----|----|------|---|
|**0**|News&Politics|2006-08-07|UCBi2mrWuNuyYy4gbM6fU18Q|abcnews|6640000|48869|978.0|2.087|LeanLeft|High|39419|19665|
|**...**|...|...|...|...|...|...|...|...|...|...|...|...|27222|

To get a first idea of the data, the visualization of the two new datasets can be found in the file [*news_channel.ipynb*](./allsides_scraping/news_channel.ipynb)


## Methods

The code used to treat raw data and create the original graph can be found in [graph_creation.ipynb](graph_construction.ipynb).
Then, [clustering.ipynb](clustering.ipynb) is used to identify communities in the graph.

Generated csv files (in `csv_output` folder) :
- channels.csv : relates channel_id to channel_num (all channels in News&Politics)
- medias.csv : relates channel_id, channel name and channel_num, for AllSides medias
- display_id_to_channels.csv : connects, for all News&Politics channels, display_id to channels_num
- authors_to_channels.csv : for all News&Politics channels, 1 line : this author has comented (at least once) this channel (identified by channel_num)
- graph_medias.csv: all graph edges, connecting two channels of AllTimes medias that have been commented by the same authors. Weight : number of authors who commented both channels
- graph_channels.csv: all graph edges, connecting two channels of News&Politics that have been commented by the same authors. Weight : number of authors who commented both channels
- louvain_filtered_graph.csv: same as graph_channels, but with non-english channels filtered out. 


### Data handling
Since this dataset is huge (~111GB compressed), it is very important to have a proper methodology to store, access, preprocess and filter it efficiently. All the data is stored locally, with the largest files on an external hard drive. The largest files are read by chunks to produce the graph. However, once the graph is done, the amount of data is much smaller (~30KB) and can be handled very easily.


### Clustering
We wanted to see if we were able to detect channel communities within our dataset. To do this we generated an undirected weigthed graph using user comments with the following methodology : If a user wrote a comment on a video of channel 1 and a comment on a video on channel 2, we create an edge of weight 1 between channel 1 and channel 2. To rapidly visualize how well it works we used `Gephi`, an open source graph visualisation platform. We discovered the louvain algorithm on gephi and realised it detected communities quite well. There were a lot of news channel that were pakistani or indian. When we ran the louvain algorithm with python using the `networkx` package, we quickly identified all the channels that were unnecessary. We then obtained a list of channels that were only in the USA and in the correct language. We ran the louvain algorithm again on our new filtered graph and obtained promising results. We indeed discovered discovered left and right-oriented communities 
<img src="./pictures/Graph.png" alt="media_bias" class="center"/>



## Proposed timeline
- 18 November: Milestone P2 due
- 25 November: Apply everything we tested yet (Louvain algorithm, graph visualisation) to the whole/larger subset of comments
- 02 December: Homework 2 due. No project work
- 09 December: Creation of the website to have an idea (even if empty for the moment), look into words occurences analysis of video titles and tags, give political score to authors, identify communities and their links.
- 16 December: Finish the data procesing work. Answer research questions, create insightful graphs and other helpful data visualizations.
- 23 December: Milestone P3 due. Focus on writing up the report or the data story, preparing the final presentation, formating for the website and the data story, 


## Team organization

- Loïc: Find the center of interest of the various communities found in the graph.
- Camille: Analyse words occurences in video tags and titles, create the website for final presentation of results
- Michel : Implement properly the clustering algorithms, colorize the graph as function of the communities, Insert the main channel labels directly on the graph.
- Stéphane: Create the graph on a larger number of comments (possibly the whole dataset). Implement the p-score to analyze the distribution of users.

[1]: https://www.nber.org/papers/w26669
[2]: https://theconversation.com/mapping-brazils-political-polarization-online-96434
[3]: https://www.allsides.com/media-bias
[4]: https://www.allsides.com
[5]: https://en.wikipedia.org/wiki/AllSides
