# %%

'''
File name: titles_analysis.py
Author: Camille Bernelin
Date created: 02/11/2022
Date last modified: 23/12/2022
Python Version: 3.9.13
'''

############# Parameters to define for further processing of videos titles

##### RUN FOR PERIOD :

year_start = "2018"
month_start = "12"
year_end = "2019"
month_end = "01"

##### FOCUS ON A WORD

word = ''

##### HOW MANY COMMUNITIES ?

nb_communities = 6

# %%


import pandas as pd
import numpy as np
import spacy
import codecs


import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import math
import os
import glob

# Word occurences count
from collections import Counter

# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dynamic graphs
import plotly.graph_objects as go
from plotly.colors import n_colors
import plotly.express as px

# Emojis management
def is_emoji(s):
    emojis = "😘◼️•🔴🤾🎅😂🚒👨🤦" # add more emojis here
    count = 0
    for emoji in emojis:
        count += s.count(emoji)
        if count > 1:
            return False
    return bool(count)

DIR = "data/"
DIR_LARGE = "data/large/"
DIR_OUT = "csv_outputs/"
PATH_METADATA = DIR_LARGE + "yt_metadata_en.jsonl.gz"

# %%

selected_commu = 1

communities = range(6)

############# Data used for processing of titles and tags

nlp = spacy.load('en_core_web_sm')
nlp.max_length=1000000

# Preprocessing
#undesired_expression_list = ["Fox News", "New York Times","NBC News"]
undesired_expression_list = []

# Filtering
start_date = year_start + "-" + month_start + "-01"
period_start = pd.to_datetime(start_date, format='%Y-%m-%d')
end_date = year_end + "-" + month_end + "-31"
period_end =pd.to_datetime(end_date, format='%Y-%m-%d')

# Processing
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
#news_lexical_field = ['news','News','wrap','Day','day','TV','Channel','channel']
news_lexical_field = ['News','news','man','Man','woman','Woman','Day','day','say','Say','says','Says','SAYS','New','new','year','Year','Call','call','LIVE','Live','live','Video','video','December','January']
my_undesired_list = ['|','l','w/','=','$',word] + news_lexical_field

named_entities = ['LOC','GPE']

# %%

############# Create a list of all videos of a selected communities, with titles and tags : use display_id_to_channels_title_tags

channels_list = pd.read_csv("csv_outputs/louvain_communities_channels_large.csv", sep=";", usecols=['channel','community'])
channels_list = pd.DataFrame(channels_list[channels_list['community']==selected_commu]['channel'])
channels_list.columns=['channel_num']
display(channels_list.head())

pd.DataFrame(columns=['title','upload_date']).to_csv(DIR_OUT+"titles_date"+str(selected_commu)+".csv", sep=';', index=False, header=True)
pd.DataFrame(columns=['tags','upload_date']).to_csv(DIR_OUT+"tags_date"+str(selected_commu)+".csv", sep=';', index=False, header=True)

for chunk in pd.read_csv(DIR_OUT + "channel_num_to_title_tags_date.csv", sep=";", chunksize=100000):
    chunk = channels_list.merge(chunk, on='channel_num')
    titles = pd.DataFrame(chunk[['title','upload_date']])
    titles.to_csv(DIR_OUT+"titles_date"+str(selected_commu)+".csv", sep=';', mode='a', index=False, header=False)
    tags = pd.DataFrame(chunk[['tags','upload_date']])
    tags.to_csv(DIR_OUT+"tags_date"+str(selected_commu)+".csv", sep=';', mode='a', index=False, header=False)

# EXEC : ~1min per community


# %%

                ######################
                ######################
                # TITLES  PROCESSING #
                ######################
                ######################

for selected_commu in communities:
    ############# Select videos whom title contains a given word

    titles = pd.read_csv(DIR_OUT + "titles_date"+str(selected_commu)+".csv",sep=";")
    display(DIR_OUT + "titles_date"+str(selected_commu)+".csv")

    titles_contain = titles[titles['title'].str.contains(word, case=False)]


    ############# Preprocess the text

    for expression in undesired_expression_list :
        titles_contain.loc['title'] = titles_contain['title'].str.replace(expression,"")

    titles_contain.to_csv(DIR_OUT + "titles_date_f.csv",sep=";",index=False, header=True)
    #titles_contain.to_csv(DIR_OUT+"titles.txt", sep="\n",index=False, header=False)

    ############# Select videos in a time period of interest

    titles = pd.read_csv(DIR_OUT + "titles_date_f.csv",sep=";")
    titles["upload_date"] = pd.to_datetime(titles["upload_date"], format='%Y-%m-%d %H:%M:%S')
    titles = titles[(titles['upload_date']>=period_start) & (titles['upload_date']<=period_end)]
    #display(titles)

    #titles['title'] = [x.encode('utf-8','ignore').decode("utf-8") for x in titles['title']]
    titles["title"].to_csv(DIR_OUT+"titles_to_process.txt", sep="\n",index=False, header=False)

    display(titles)

    ############# Process the titles 

    books = list()

    with codecs.open(os.path.join(DIR_OUT,"titles_to_process.txt"), encoding='utf8') as f:
        books.append(f.read())

    # Remove new lines
    books = [" ".join(b.split()) for b in books]
    display("Length of the book is " + str(len(books[0])) + " characters, " + str(len(books[0])/nlp.max_length) + " times the max length that can be processed at once.")

    doc_processed = []

    # Processing of titles
    for i in range(0,int(len(books[0])/nlp.max_length)+1):
        display('ITERATION ' + str(i))

        # Tokenization
        print("Tokenization")
        doc = nlp(books[0][i*nlp.max_length:(i+1)*nlp.max_length])

        # Named entities recognition
        print("Named entities recognition")
        for ent in doc.ents:
            if ent.label_ in named_entities:
                doc_processed = doc_processed + [ent.text]
            
        # Punctuation and stopwords removal
        print("Punctuation and stopwords removal")
        doc_processed = doc_processed + [token.text for token in doc if (not (token.ent_type_ in named_entities) and not token.is_digit and not token.is_stop and not token.is_punct and not (token.text in my_undesired_list) and not is_emoji(token.text))]

        # Removal of undesired characters
        doc_processed = [token.replace(',','') for token in doc_processed]
        doc_processed = [token.replace('.','') for token in doc_processed]
        doc_processed = [token.replace("'s",'') for token in doc_processed]

        print("Output")
        #doc_processed = [x.encode('latin1','ignore').decode("latin1") for x in doc_processed]
        pd.DataFrame(doc_processed).to_csv(DIR_OUT+"titles_words.csv", sep=";",index=False, header=['word'])

    # EXEC = ~2min30 per iteration (Tokenization takes 90% of exec time)
    
    ############# Count words occurences

    PATH_OUT = DIR_OUT+"communities_comparison/titles_occurences_"+str(selected_commu)+"_"+word+"_"+start_date+"_"+end_date+".csv"

    titles_processed = pd.read_csv(DIR_OUT+"titles_words.csv", sep=';')
    #titles_processed = [x.encode('utf-8','ignore').decode("utf-8") for x in titles_processed['word']]

    display(titles_processed.head)

    # Count occurences
    titles_processed_lowercase = [str(word).lower() for word in titles_processed['word']]
    word_freq = Counter(titles_processed_lowercase)
    common_words = word_freq.most_common()
    common_words_out = pd.DataFrame(common_words)
    common_words_out.columns=['word','occurences']

    #display(common_words)

    common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())

    common_words_out.to_csv(PATH_OUT,sep=';')
    display(common_words_out.head(30))

# %%

######### Comparison of communities given words ensembles

            ######################
            ######################
            ## WORDS  ENSEMBLES ##
            ######################
            ######################


# Filtering

result = []
result_df = pd.DataFrame()

trump_list = ['maga','security','trump','wall','evangelist']
national_list = ['shutdown','wall','security','shooting','pelosi','immigration','maga']
surnatural_list = ['ufo','prophecy','christ','truth','spirit','pray','prophet']
conspi_list = ['satanic','fake','truth','alien','lie','moon','9/11','pedophile']
international_list = ['brexit','eu','china','europe','macron','boris','merkel','yemen','asia','africa','india','modi','morrison']

list_of_lists = [trump_list, national_list, surnatural_list, conspi_list, international_list]
list_of_lists_name = ['trump', 'national politics', 'religion and beliefs', 'conspirationist', 'international']
nb_list = 1
list_name = list_of_lists_name[nb_list]
targets_list = list_of_lists[nb_list]

for j in range(len(targets_list)):
    temp = pd.DataFrame(columns = ['community','rank_title','rank_tag','title_frequency'])
    target = targets_list[j]
    use_tags = False

    for i in range(0,6):
        titles=pd.read_csv(DIR_OUT+"communities_comparison/titles_occurences_"+str(i)+"_"+word+"_"+start_date+"_"+end_date+".csv", sep=';')
        if use_tags :
            if (titles['word']==target).any() or (tags['word']==target).any():
                tags=pd.read_csv(DIR_OUT+"communities_comparison/tags_occurences_"+str(i)+"_"+word+"_"+start_date+"_"+end_date+".csv", sep=';')
                temp.loc[len(temp.index)] = [i, titles[titles['word']==target].index.values[0], tags[tags['word']==word].index.values[0], float(titles[titles['word']==target]['frequency'])]
            else:
                display(target + " is not in community " + str(i))
                temp.loc[len(temp.index)] = [i,math.inf, '-', 0]
        else:
            if (titles['word']==target).any():
                temp.loc[len(temp.index)] = [i, titles[titles['word']==target].index.values[0], '-', float(titles[titles['word']==target]['frequency'])]
            else:
                display(target + " is not in community " + str(i))
                temp.loc[len(temp.index)] = [i,math.inf, '-', 0]

    result = result + [temp]


############ GRAPH : Words occurences

data = pd.DataFrame()
for i in range(0,len(targets_list)):
    data.insert(i, column=targets_list[i], value=result[i]['title_frequency'])

#display(data)

fig = px.line(data, title='Words occurences in different communities : '+list_name+' topics',labels={'value':'frequency','index':'community','variable':'Topics :'},width=600, height=400)
not_active_traces = [targets_list[i] for i in range(len(targets_list)) if i>len(targets_list)*1]
fig.for_each_trace(lambda trace: trace.update(visible='legendonly') if trace.name in not_active_traces else ())

list_of_lists_name = [word.replace(' ','_') for word in list_of_lists_name]
list_name = list_of_lists_name[nb_list]

fig.show()
fig.write_html("figures/words_occurences_"+list_name+".html")

display("figures/words_occurences_"+list_name+".html")
#plt.show(sns.lineplot(data=data))

# %%

            ######################
            ######################
            ## CLOSENESS MATRIX ##
            ######################
            ######################

############ GRAPH : Closeness matrix

closeness_matrix = pd.DataFrame(index=communities,columns=communities)

for x in communities:
    for y in communities:
        for target in range(len(targets_list)):
            a = result[target][result[target]['community']==x]['title_frequency']
            b = result[target][result[target]['community']==y]['title_frequency']
            closeness_matrix[x][y]=int(abs(float(a)-float(b))*100000)

display(closeness_matrix)

colors = n_colors('rgb(0, 180, 0)', 'rgb(30, 30, 30)',closeness_matrix.max().max()+1, colortype='rgb')
a = np.stack(closeness_matrix[0].to_numpy()).astype(int)
b = np.stack(closeness_matrix[1].to_numpy()).astype(int)
c = np.stack(closeness_matrix[2].to_numpy()).astype(int)
d = np.stack(closeness_matrix[3].to_numpy()).astype(int)
e = np.stack(closeness_matrix[4].to_numpy()).astype(int)
f = np.stack(closeness_matrix[5].to_numpy()).astype(int)

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>0</b>','<b>1</b>','<b>2</b>','<b>3</b>','<b>4</b>','<b>5</b>'],
    line_color='white', fill_color='white',
    align='center',font=dict(color='black', size=12)
  ),
  cells=dict(
    values=[a,b,c,d,e,f],
    line_color=[np.array(colors)[a], np.array(colors)[b], np.array(colors)[c], np.array(colors)[d], np.array(colors)[e], np.array(colors)[f]],
    fill_color=[np.array(colors)[a], np.array(colors)[b], np.array(colors)[c], np.array(colors)[d], np.array(colors)[e], np.array(colors)[f]],
    align='center', font=dict(color='white', size=11)
    ))
])

fig.show()
fig.write_html("figures/closeness_matrix.html")

# %%

            ######################
            ######################
            #### 60 MOST CITED ###
            ######################
            ######################

# Using seaborn

sns.set(font_scale=1.5)
fig, ax = plt.subplots(3, 2, figsize=(35, 55))

palette = sns.color_palette("coolwarm", n_colors = 100)
palette.reverse()

for community in communities:
    position_y = int(community/2)
    position_x = community % 2
    
    data=pd.read_csv(DIR_OUT+"communities_comparison/titles_occurences_"+str(community)+"_"+word+"_"+start_date+"_"+end_date+".csv", sep=';')
    data=data.head(30).sort_values(by='occurences', ascending=False)
    sns.barplot(x = 'frequency', y = 'word', data=data, ax = ax[position_y,position_x], palette=palette)
    ax[position_y,position_x].set_title("Most cited topics in community "+str(community), fontsize = 25)
    ax[position_y,position_x].set_xlabel("Frequency", fontsize = 20)
    ax[position_y,position_x].set_ylabel("Topics", fontsize = 20)

plt.savefig("figures/20_most_cited_topics")

# %%

# Using plotly

# Read previously computed files

data = pd.DataFrame()

for community in communities:
    temp = pd.read_csv(DIR_OUT+"communities_comparison/titles_occurences_"+str(community)+"_"+word+"_"+start_date+"_"+end_date+".csv", sep=';',index_col=0)
    myarray = np.empty(len(temp), dtype=int)
    myarray.fill(community)
    temp.insert(0,column='community',value=myarray)
    temp = temp.sort_values(by='frequency',ascending=True)
    temp = temp[len(temp)-30:len(temp)]
    
    data = pd.concat([data, temp])

display(data)

# Create histogram

fig = px.histogram(data,
                   x="frequency",
                   y="word",
                   title='Histogram of polarization scores',
                   animation_frame="community",
                   opacity=0.8,
                   range_x=[0,0.035],
                   orientation='h')

fig.update_layout(autosize=False,
                  width=700,
                  height=800,
                  title={'text' : 'Histogram of topics occurence frequency',
                         'x':0.5,
                         'xanchor': 'center'
                         },
                  xaxis_title="Frequency",
                  yaxis_title=None,
                  margin=dict(l=100,r=50,b=100,t=100,pad=4))

fig["layout"].pop("updatemenus")
fig.update_yaxes(automargin=False, ticklabelposition="outside",position=0)
fig.update_yaxes(tickfont_size=14)
fig.write_html("figures/60_most_cited_topics.html")

fig.show()

# %%


            ######################
            ######################
            ######## LOCO ########
            ####### DATASET ######
            ######################
            ######################


# %%

######## Creating a smaller copy of the LOCO.json file

# chunk = 10 000 000   ->  9 766 KB
# total file : chunk = 587 622 363

text = ""
chunk = 590000000

with codecs.open("data/LOCO.json", 'r', encoding='latin1') as file:
    #writing the header
    text = file.read(chunk)
file.close()

display(text)

with codecs.open("data/LOCO_small.txt", 'w', encoding='latin1') as file:
    file.write(text)
file.close()

# %% 
######## Exctracting titles in LOCO

result = ""
text = ""
unfinished_title = False
nb_titles = 0
chunk = 1000000

with codecs.open("data/LOCO_small.txt", 'r', encoding = 'latin1') as file:

    #writing the header
    text = file.read(chunk)

    while text:

        if unfinished_title:
            result = result +  text[0:text.find('"txt"')-2]
            text = text[text.find('txt')+8:len(text)]
            unfinished_title = False

        while text.find('"title"') != -1:
            nb_titles = nb_titles + 1
            text = text[text.find('"title"')+9:len(text)]

            if text.find('"txt"') != -1:
                result = result +  text[0:text.find('"txt"')-2] + "\n"
                text = text[text.find('"txt"')+7:len(text)]
            else:
                unfinished_title = True

        text = file.read(chunk)

file.close()

with codecs.open("data/LOCO_titles.txt", 'w', encoding = 'latin1') as file:
    file.write(result)
file.close()

display(nb_titles)


# %%

######## Processing loco just as other books

loco = ""
doc_processed = []

with codecs.open("data/LOCO_titles.txt", 'r', encoding = 'latin1') as file:
    loco = file.read()
file.close()


display("Length of the book is " + str(len(loco)) + " characters, " + str(len(loco)/nlp.max_length) + " times the max length that can be processed at once.")

# Processing
for i in range(0,int(len(loco)/nlp.max_length)+1):
    display('ITERATION ' + str(i))

    # Tokenization
    print("Tokenization")
    doc = nlp(loco[i*nlp.max_length:(i+1)*nlp.max_length])

    # Named entities recognition
    print("Named entities recognition")
    for ent in doc.ents:
        if ent.label_ in named_entities:
            doc_processed = doc_processed + [ent.text]
        
    # Punctuation and stopwords removal
    print("Punctuation and stopwords removal")
    doc_processed = doc_processed + [token.text for token in doc if (not (token.ent_type_ in named_entities) and not token.is_digit and not token.is_stop and not token.is_punct and not (token.text in my_undesired_list) and not is_emoji(token.text))]

    # Removal of undesired characters
    doc_processed = [token.replace(',','') for token in doc_processed]
    doc_processed = [token.replace('.','') for token in doc_processed]
    doc_processed = [token.replace("'s",'') for token in doc_processed]

    doc_processed = [token_text for token_text in doc_processed if not token_text=='']

    print("Output")
    #doc_processed = [x.encode('latin1','ignore').decode("latin1") for x in doc_processed]
    pd.DataFrame(doc_processed).to_csv(DIR_OUT+"loco_processed.csv", sep=";",index=False, header=['word'], encoding='latin1')


# %%

###### Computing occurences count for LOCO

loco_processed = pd.read_csv(DIR_OUT+"loco_processed.csv", sep=';', encoding='latin1')

display(loco_processed.head)

# Count occurences
loco_processed_lowercase = [str(word).lower() for word in loco_processed['word']]
word_freq = Counter(loco_processed_lowercase)
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

# Correction due to abscence of covid in studied years
# divide by 10
mask = (common_words_out['word'].isin(['coronavirus','vaccine','vaccines','pandemic','virus','pharma']))
common_words_out_valid = common_words_out[mask]
common_words_out.loc[mask, 'occurences'] = common_words_out.loc[mask, 'occurences']/10
# delete topics
mask = (common_words_out['word'].isin(['covid-19']))
mask[[7,50,160,300,436]]=True
common_words_out = common_words_out.drop(common_words_out[mask].index,axis="index")

common_words_out = common_words_out.drop(common_words_out.loc[2000:len(common_words_out)].index, axis='index')

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())
common_words_out = common_words_out.sort_values(by='frequency', ascending = False)[['word','frequency']]

common_words_out.to_csv(DIR_OUT+"loco_occurences.csv")
display(common_words_out.head(10))

# %%

###### Comparing word occurences in LOCO and in our titles for a given

loco_occurences = pd.read_csv(DIR_OUT+"loco_occurences.csv", index_col = 0)
distance_loco = []

# Merge with communities occurences counts
for selected_commu in communities:

    PATH_COMMU = DIR_OUT+"communities_comparison/titles_occurences_"+str(selected_commu)+"_"+word+"_"+start_date+"_"+end_date+".csv"
    titles_occurences = pd.read_csv(PATH_COMMU, sep=';',index_col=0,usecols=['word','frequency'])
    loco_occurences.columns=['word','frequency_loco']

    # Merge to identify common words
    merged = titles_occurences.merge(loco_occurences, on='word', how='outer').fillna(0)

    # Normalize both vectors, not to be influenced by the number of words in the community
    merged['frequency'] = merged['frequency'] / merged['frequency'].abs().max()
    merged['frequency_loco'] = merged['frequency_loco'] / merged['frequency_loco'].abs().max()

    # Apply a logarithmic scale
    merged.insert(1,column='log_frequency',value=np.log10(merged['frequency']+1))
    merged.insert(1,column='log_frequency_loco',value=np.log10(merged['frequency_loco']+1))

    # Compute a 2-norm distance
    merged.insert(1,column='distance',value=merged['log_frequency']-merged['log_frequency_loco'])
    merged.insert(1,column='squared_distance',value=merged['distance'].pow(2))

    distance_loco = distance_loco + [math.sqrt(merged['squared_distance'].sum())]
m = 0.0
for x in distance_loco:
    if x>m:
        m=x
distance_loco = [m-x for x in distance_loco]

# Display result


display(distance_loco)

fig = px.line(distance_loco, title='Distance between topics in a community and topics in conspiracist medias',labels={'index':'community','value':'distance'},width=650, height=400)
fig.update_traces(showlegend=False)
fig.show()


# %%


            ######################
            ######################
            #### BLACK VOICES ####
            ####### DATASET ######
            ######################
            ######################

# %%

######## Dataset about black voices

# Exctract titles and descriptions

dataset = pd.read_json("data/News_Category_Dataset_v3.json", lines = True)

dataset_black = dataset[dataset['category']=='BLACK VOICES'][['headline','short_description']]
dataset_black.to_csv("data/News_black_voices_titles.txt", sep='\n', index=None, header=None)

display(dataset_black)

# %%

######## Processing black_voices just as other books

black_voices = ""
doc_processed = []

with codecs.open("data/News_black_voices_titles.txt", 'r') as file:
    black_voices = file.read()
file.close()


display("Length of the book is " + str(len(black_voices)) + " characters, " + str(len(black_voices)/nlp.max_length) + " times the max length that can be processed at once.")

# Processing
for i in range(0,int(len(black_voices)/nlp.max_length)+1):
    display('ITERATION ' + str(i))

    # Tokenization
    print("Tokenization")
    doc = nlp(black_voices[i*nlp.max_length:(i+1)*nlp.max_length])

    # Named entities recognition
    print("Named entities recognition")
    for ent in doc.ents:
        if ent.label_ in named_entities:
            doc_processed = doc_processed + [ent.text]
        
    # Punctuation and stopwords removal
    print("Punctuation and stopwords removal")
    doc_processed = doc_processed + [token.text for token in doc if (not (token.ent_type_ in named_entities) and not token.is_digit and not token.is_stop and not token.is_punct and not (token.text in my_undesired_list) and not is_emoji(token.text))]

# Removal of undesired characters
doc_processed = [token.replace(',','') for token in doc_processed]
doc_processed = [token.replace('.','') for token in doc_processed]
doc_processed = [token.replace("'s",'') for token in doc_processed]

doc_processed = [token_text for token_text in doc_processed if not token_text=='']

print("Output")
#doc_processed = [x.encode('latin1','ignore').decode("latin1") for x in doc_processed]
pd.DataFrame(doc_processed).to_csv(DIR_OUT+"black_voices_processed.csv", sep=";",index=False, header=['word'])

# %%

###### Computing occurences count for black_voices

black_voices_processed = pd.read_csv(DIR_OUT+"black_voices_processed.csv", sep=';', encoding='latin1')

display(black_voices_processed.head)

# Count occurences
black_voices_processed_lowercase = [str(word).lower() for word in black_voices_processed['word']]
word_freq = Counter(black_voices_processed_lowercase)
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

# Correction due to abscence of covid in studied years
# divide by 10
mask = (common_words_out['word'].isin(['coronavirus','vaccine','vaccines','pandemic','virus','pharma']))
common_words_out_valid = common_words_out[mask]
common_words_out.loc[mask, 'occurences'] = common_words_out.loc[mask, 'occurences']/10
# delete topics
mask = (common_words_out['word'].isin(['covid-19']))
mask[[7,50,160,300,436]]=True
common_words_out = common_words_out.drop(common_words_out[mask].index,axis="index")

common_words_out = common_words_out.drop(common_words_out.loc[100:len(common_words_out)].index, axis='index')

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())
common_words_out = common_words_out.sort_values(by='frequency', ascending = False)[['word','frequency']]

common_words_out.to_csv(DIR_OUT+"black_voices_occurences.csv")

display(common_words_out.head(10))

# %% 

###### Comparing word occurences in black_voices and in our titles for a given

black_voices_occurences = pd.read_csv(DIR_OUT+"black_voices_occurences.csv", index_col = 0)
distance_black = []

# Merge with communities occurences counts
for selected_commu in communities:

    PATH_COMMU = DIR_OUT+"communities_comparison/titles_occurences_"+str(selected_commu)+"_"+word+"_"+start_date+"_"+end_date+".csv"

    titles_occurences = pd.read_csv(PATH_COMMU, sep=';',index_col=0,usecols=['word','frequency'])
    black_voices_occurences.columns=['word','frequency_black_voices']

    # Merge to identify common words
    merged = titles_occurences.merge(black_voices_occurences, on='word', how='inner').fillna(0)

    # Normalize both vectors, not to be influenced by the number of words in the community
    merged['frequency'] = merged['frequency'] / merged['frequency'].abs().max()
    merged['frequency_black_voices'] = merged['frequency_black_voices'] / merged['frequency_black_voices'].abs().max()

    # Apply a logarithmic scale
    merged.insert(1,column='log_frequency',value=np.log10(merged['frequency']+1))
    merged.insert(1,column='log_frequency_black_voices',value=np.log10(merged['frequency_black_voices']+1))

    # Compute a 2-norm distance
    merged.insert(1,column='distance',value=merged['log_frequency']-merged['log_frequency_black_voices'])
    merged.insert(1,column='squared_distance',value=merged['distance'].pow(2))

    distance_black = distance_black + [math.sqrt(merged['squared_distance'].sum())]
m = 0.0
for x in distance_black:
    if x>m:
        m=x
distance_black = [m-x for x in distance_black]

# Display result

display(distance_black)

fig = px.line(distance_black, title='Distance between topics in a community and topics in medias sharing black voices',labels={'index':'community','value':'distance'},width=650, height=400)
fig.update_traces(showlegend=False)
fig.show()

# %%


            ######################
            ######################
            #### QUEER VOICES ####
            ####### DATASET ######
            ######################
            ######################

# %%

######## Dataset about queer voices

# Exctract titles and descriptions

dataset = pd.read_json("data/News_Category_Dataset_v3.json", lines = True)

dataset_queer = dataset[dataset['category']=='QUEER VOICES'][['headline','short_description']]
dataset_queer.to_csv("data/News_queer_voices_titles.txt", sep='\n', index=None, header=None)

display(dataset_queer)

# %%

######## Processing queer_voices just as other books

queer_voices = ""
doc_processed = []

with codecs.open("data/News_queer_voices_titles.txt", 'r') as file:
    queer_voices = file.read()
file.close()


display("Length of the book is " + str(len(queer_voices)) + " characters, " + str(len(queer_voices)/nlp.max_length) + " times the max length that can be processed at once.")

# Processing
for i in range(0,int(len(queer_voices)/nlp.max_length)+1):
    display('ITERATION ' + str(i))

    # Tokenization
    print("Tokenization")
    doc = nlp(queer_voices[i*nlp.max_length:(i+1)*nlp.max_length])

    # Named entities recognition
    print("Named entities recognition")
    for ent in doc.ents:
        if ent.label_ in named_entities:
            doc_processed = doc_processed + [ent.text]
        
    # Punctuation and stopwords removal
    print("Punctuation and stopwords removal")
    doc_processed = doc_processed + [token.text for token in doc if (not (token.ent_type_ in named_entities) and not token.is_digit and not token.is_stop and not token.is_punct and not (token.text in my_undesired_list) and not is_emoji(token.text))]

# Removal of undesired characters
doc_processed = [token.replace(',','') for token in doc_processed]
doc_processed = [token.replace('.','') for token in doc_processed]
doc_processed = [token.replace("'s",'') for token in doc_processed]

doc_processed = [token_text for token_text in doc_processed if not token_text=='']

print("Output")
#doc_processed = [x.encode('latin1','ignore').decode("latin1") for x in doc_processed]
pd.DataFrame(doc_processed).to_csv(DIR_OUT+"queer_voices_processed.csv", sep=";",index=False, header=['word'])

# %%

###### Computing occurences count for queer_voices

queer_voices_processed = pd.read_csv(DIR_OUT+"queer_voices_processed.csv", sep=';', encoding='latin1')

display(queer_voices_processed.head)

# Count occurences
queer_voices_processed_lowercase = [str(word).lower() for word in queer_voices_processed['word']]
word_freq = Counter(queer_voices_processed_lowercase)
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

# Correction due to abscence of covid in studied years
# divide by 10
mask = (common_words_out['word'].isin(['coronavirus','vaccine','vaccines','pandemic','virus','pharma']))
common_words_out_valid = common_words_out[mask]
common_words_out.loc[mask, 'occurences'] = common_words_out.loc[mask, 'occurences']/10
# delete topics
mask = (common_words_out['word'].isin(['covid-19']))
mask[[7,50,160,300,436]]=True
common_words_out = common_words_out.drop(common_words_out[mask].index,axis="index")

common_words_out = common_words_out.drop(common_words_out.loc[100:len(common_words_out)].index, axis='index')

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())
common_words_out = common_words_out.sort_values(by='frequency', ascending = False)[['word','frequency']]

common_words_out.to_csv(DIR_OUT+"queer_voices_occurences.csv")

display(common_words_out.head(10))

# %% 

###### Comparing word occurences in queer_voices and in our titles for a given

queer_voices_occurences = pd.read_csv(DIR_OUT+"queer_voices_occurences.csv", index_col = 0)
distance_queer = []

# Merge with communities occurences counts
for selected_commu in communities:

    PATH_COMMU = DIR_OUT+"communities_comparison/titles_occurences_"+str(selected_commu)+"_"+word+"_"+start_date+"_"+end_date+".csv"
    titles_occurences = pd.read_csv(PATH_COMMU, sep=';',index_col=0,usecols=['word','frequency'])
    queer_voices_occurences.columns=['word','frequency_queer_voices']

    # Merge to identify common words
    merged = titles_occurences.merge(queer_voices_occurences, on='word', how='inner').fillna(0)

    # Normalize both vectors, not to be influenced by the number of words in the community
    merged['frequency'] = merged['frequency'] / merged['frequency'].abs().max()
    merged['frequency_queer_voices'] = merged['frequency_queer_voices'] / merged['frequency_queer_voices'].abs().max()

    # Apply a logarithmic scale
    merged.insert(1,column='log_frequency',value=np.log10(merged['frequency']+1))
    merged.insert(1,column='log_frequency_queer_voices',value=np.log10(merged['frequency_queer_voices']+1))

    # Compute a 2-norm distance
    merged.insert(1,column='distance',value=merged['log_frequency']-merged['log_frequency_queer_voices'])
    merged.insert(1,column='squared_distance',value=merged['distance'].pow(2))

    distance_queer = distance_queer + [math.sqrt(merged['squared_distance'].sum())]

m = 0.0
for x in distance_queer:
    if x>m:
        m=x
distance_queer = [m-x for x in distance_queer]

# Display result

display(distance_queer)

fig = px.line(distance_queer, title='Distance between topics in a community and topics in medias sharing queer voices',labels={'index':'community','value':'distance'},width=650, height=400)
fig.update_traces(showlegend=False)
fig.show()

# %%


            ######################
            ######################
            #### CLIMATE NEWS ####
            ####### DATASET ######
            ######################
            ######################

# %%

######## Dataset about climate news

dataset_climate = pd.DataFrame()

# Exctract titles and descriptions
path = "data/climate-news-db-dataset/articles"
csv_files = glob.glob(os.path.join(path, "*.jsonlines"))
for f in csv_files:
    display(f)
    dataset_climate = pd.concat([dataset_climate,pd.read_json(f, lines = True)], axis=0)

dataset_climate = dataset_climate['headline']
dataset_climate.to_csv("data/News_climate_voices_titles.txt", sep='\n', index=None, header=None)

display(dataset_climate)

# %%

######## Processing climate_voices just as other books

climate_voices = ""
doc_processed = []

with codecs.open("data/News_climate_voices_titles.txt", 'r',encoding='latin1') as file:
    climate_voices = file.read()
file.close()


display("Length of the book is " + str(len(climate_voices)) + " characters, " + str(len(climate_voices)/nlp.max_length) + " times the max length that can be processed at once.")

# Processing
for i in range(0,int(len(climate_voices)/nlp.max_length)+1):
    display('ITERATION ' + str(i))

    # Tokenization
    print("Tokenization")
    doc = nlp(climate_voices[i*nlp.max_length:(i+1)*nlp.max_length])

    # Named entities recognition
    print("Named entities recognition")
    for ent in doc.ents:
        if ent.label_ in named_entities:
            doc_processed = doc_processed + [ent.text]
        
    # Punctuation and stopwords removal
    print("Punctuation and stopwords removal")
    doc_processed = doc_processed + [token.text for token in doc if (not (token.ent_type_ in named_entities) and not token.is_digit and not token.is_stop and not token.is_punct and not (token.text in my_undesired_list) and not is_emoji(token.text))]

# Removal of undesired characters
doc_processed = [token.replace(',','') for token in doc_processed]
doc_processed = [token.replace('.','') for token in doc_processed]
doc_processed = [token.replace("'s",'') for token in doc_processed]

doc_processed = [token_text for token_text in doc_processed if not token_text=='']

print("Output")
#doc_processed = [x.encode('latin1','ignore').decode("latin1") for x in doc_processed]
pd.DataFrame(doc_processed).to_csv(DIR_OUT+"climate_voices_processed.csv", sep=";",index=False, header=['word'])

# %%

###### Computing occurences count for climate_voices

climate_voices_processed = pd.read_csv(DIR_OUT+"climate_voices_processed.csv", sep=';', encoding='latin1')

display(climate_voices_processed.head)

# Count occurences
climate_voices_processed_lowercase = [str(word).lower() for word in climate_voices_processed['word']]
word_freq = Counter(climate_voices_processed_lowercase)
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

# Delete errors
mask = (common_words_out['word'].isin(['\r\n']))
common_words_out = common_words_out.drop(common_words_out[mask].index,axis="index")

common_words_out = common_words_out.drop(common_words_out.loc[1000:len(common_words_out)].index, axis='index')

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())
common_words_out = common_words_out.sort_values(by='frequency', ascending = False)[['word','frequency']]

common_words_out.to_csv(DIR_OUT+"climate_voices_occurences.csv")

display(common_words_out.head(10))

# %% 

###### Comparing word occurences in climate_voices and in our titles for a given

climate_voices_occurences = pd.read_csv(DIR_OUT+"climate_voices_occurences.csv", index_col = 0)
distance_climate = []

# Merge with communities occurences counts
for selected_commu in communities:

    PATH_COMMU = DIR_OUT+"communities_comparison/titles_occurences_"+str(selected_commu)+"_"+word+"_"+start_date+"_"+end_date+".csv"
    titles_occurences = pd.read_csv(PATH_COMMU, sep=';',index_col=0,usecols=['word','frequency'])
    climate_voices_occurences.columns=['word','frequency_climate_voices']

    # Merge to identify common words
    merged = titles_occurences.merge(climate_voices_occurences, on='word', how='inner').fillna(0)

    # Normalize both vectors, not to be influenced by the number of words in the community
    merged['frequency'] = merged['frequency'] / merged['frequency'].abs().max()
    merged['frequency_climate_voices'] = merged['frequency_climate_voices'] / merged['frequency_climate_voices'].abs().max()

    # Apply a logarithmic scale
    merged.insert(1,column='log_frequency',value=np.log10(merged['frequency']+1))
    merged.insert(1,column='log_frequency_climate_voices',value=np.log10(merged['frequency_climate_voices']+1))

    # Compute a 2-norm distance
    merged.insert(1,column='distance',value=merged['log_frequency']-merged['log_frequency_climate_voices'])
    merged.insert(1,column='squared_distance',value=merged['distance'].pow(2))

    distance_climate = distance_climate + [math.sqrt(merged['squared_distance'].sum())]

m = 0.0
for x in distance_climate:
    if x>m:
        m=x
distance_climate = [m-x for x in distance_climate]


display(distance_climate)

fig = px.line(distance_climate, title='Distance between topics in a community and topics in medias sharing climate news',labels={'index':'community','value':'distance'},width=650, height=400)
fig.update_traces(showlegend=False)
fig.show()

# %%

########### HEAT MAP OF DISTANCES

# Normalization
distance_queer = [element/sum(distance_queer) for element in distance_queer]
distance_black = [element/sum(distance_black) for element in distance_black]
distance_loco = [element/sum(distance_loco) for element in distance_loco]
#distance_climate = [element/sum(distance_climate) for element in distance_climate]

fig = px.imshow([distance_loco,distance_black,distance_queer],
                title="Distance between selected datasets and titles of videos in our communities,<br>\t in terms of frequency of occurence of words",
                labels=dict(x="Community", y="Lexical field" , color="Distance"),
                x=['0','1','2','3','4','5'], 
                y=['Conspiracy','Black voices','Queer voices',],
                width=700, height=400)
fig.show()
fig.write_html("figures/heat_map_datasets.html")

# %%

            ######################
            ######################
            ### GATHERING URLS ###
            ######################
            ######################

# %%

df = pd.read_csv('data/df_channels_en.tsv.gz', sep='\t', compression='infer', usecols=['channel','name_cc','subscribers_cc','videos_cc'], index_col = 0)
display(df.head)
channel_num = pd.read_csv('csv_outputs/channels.csv', sep=';')
channel_num.columns=['channel_num','channel']
display(channel_num.head)
df = df.merge(channel_num, on='channel')
df['channel'] = 'https://www.youtube.com/channel/' + df['channel'].astype(str)
display(df.head(10))

communities_list = pd.read_csv("csv_outputs/louvain_communities_channels_large.csv", sep=";", usecols=['channel','community'])
communities_list.columns=['channel_num','community']

# %%

df_commu = df.merge(communities_list, on='channel_num')
result = []

for community in communities:
    temp = df_commu[df_commu['community']==community]
    temp = temp.sort_values(by='subscribers_cc')

    temp=temp[['community','subscribers_cc','name_cc','channel']]
    temp.head(10).to_csv('csv_outputs/communities_description.csv', sep=';',mode='a',header=None, index=None)
    display(community)
    display(temp.head(10))