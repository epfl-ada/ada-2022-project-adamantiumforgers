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

# Word occurences count
from collections import Counter

# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dynamic graphs
import plotly.graph_objects as go
from plotly.colors import n_colors
import plotly.express as px
#from plotly.subplots import make_subplots

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
word = ''
start_date = "2018-12-01"
period_start = pd.to_datetime(start_date, format='%Y-%m-%d')
end_date = "2019-01-31"
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

# EXEC : ~1min

# %%

            ######################
            ######################
            ####### TITLES #######
            ######################
            ######################

# %%

############# Select videos whom title contains a given word

titles = pd.read_csv(DIR_OUT + "titles_date"+str(selected_commu)+".csv",sep=";")

titles_contain = titles[titles['title'].str.contains(word, case=False)]
display(titles_contain)

############# Preprocess the text

for expression in undesired_expression_list :
    titles_contain.loc['title'] = titles_contain['title'].str.replace(expression,"")


titles_contain.to_csv(DIR_OUT + "titles_date_f.csv",sep=";",index=False, header=True)
#titles_contain.to_csv(DIR_OUT+"titles.txt", sep="\n",index=False, header=False)


# %%

############# Select videos in a time period of interest

titles = pd.read_csv(DIR_OUT + "titles_date_f.csv",sep=";")
titles["upload_date"] = pd.to_datetime(titles["upload_date"], format='%Y-%m-%d %H:%M:%S')
titles = titles[(titles['upload_date']>=period_start) & (titles['upload_date']<=period_end)]
#display(titles)

#titles['title'] = [x.encode('utf-8','ignore').decode("utf-8") for x in titles['title']]
titles["title"].to_csv(DIR_OUT+"titles_to_process.txt", sep="\n",index=False, header=False)

# %%

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

# %%

############# Analyze sentiment

analyzer = SentimentIntensityAnalyzer()

positive_sent = []
#iterate through the sentences, get polarity scores, choose a value
[positive_sent.append(analyzer.polarity_scores(sent.text)['pos']) for sent in doc.sents]
display("positive : " + str(sum(positive_sent)/len(positive_sent)))

negative_sent = []
[negative_sent.append(analyzer.polarity_scores(sent.text)['neg']) for sent in doc.sents]
display("negative : " + str(sum(negative_sent)/len(negative_sent)))


# %%

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

            ######################
            ######################
            ####### TAGS #########
            ######################
            ######################

############# Select videos whom tag contains a given word

tags = pd.read_csv(DIR_OUT + "tags_date"+str(selected_commu)+".csv",sep=";")

tags = tags[~(tags['tags'].isnull())]

tags_contain = tags[tags['tags'].str.contains(word, case=False)]
tags_contain.to_csv(DIR_OUT + "tags_date_f.csv",sep=";",index=False, header=True)

display(tags_contain)

# %%

############# Select videos in a time period of interest

tags = pd.read_csv(DIR_OUT + "tags_date_f.csv",sep=";")
tags["upload_date"] = pd.to_datetime(tags["upload_date"], format='%Y-%m-%d %H:%M:%S')
tags = tags[(tags['upload_date']>=period_start) & (tags['upload_date']<=period_end)]
display(tags)

tags['tags'] = [x.encode('latin1','ignore').decode("latin1") for x in tags['tags']]
tags["tags"].to_csv(DIR_OUT+"tags_to_process.txt", sep="\n",index=False, header=False)

# %%

############# Process the tags 

books = list()

with codecs.open(os.path.join(DIR_OUT,"tags_to_process.txt"), encoding='utf-8') as f:
    books.append(f.read())

# Remove new lines
books = [" ".join(b.split()) for b in books]
display("Length of the book is " + str(len(books[0])/nlp.max_length) + " times the max length that can be processed at once.")

# Remove stopwords and punctuation
for i in range(0,int(len(books[0])/nlp.max_length)+1):
    display('iteration ' + str(i))
    doc = nlp(books[0][i*nlp.max_length:(i+1)*nlp.max_length])

    doc_processed = [token.text for token in doc if (not token.is_digit and not token.is_stop and not token.is_punct and not (token.text in my_undesired_list))]
    doc_processed = [token.replace(',','') for token in doc_processed]
    #doc_processed = [x.encode('utf-8','ignore').decode("utf-8") for x in doc_processed]

    with open(DIR_OUT+"tags_words.csv", 'w') as file:
        #writing the header
        file.write('word\n')
        # writing the data
        for x in doc_processed:
            file.write(x+'\n')
    file.close()

# EXEC = ~2min30 per 100'000 characters

# %%

############# Analyze sentiment

analyzer = SentimentIntensityAnalyzer()

positive_sent = []
#iterate through the sentences, get polarity scores, choose a value
[positive_sent.append(analyzer.polarity_scores(sent.text)['pos']) for sent in doc.sents]
display("positive : " + str(sum(positive_sent)/len(positive_sent)))

negative_sent = []
[negative_sent.append(analyzer.polarity_scores(sent.text)['neg']) for sent in doc.sents]
display("negative : " + str(sum(negative_sent)/len(negative_sent)))


# %%

############# Count words occurences

PATH_OUT = DIR_OUT+"communities_comparison/tags_occurences_"+str(selected_commu)+"_"+word+"_"+start_date+"_"+end_date+".csv"

tags_processed = pd.read_csv(DIR_OUT+"tags_words.csv", sep=',', encoding='latin1')
#tags_processed = [x.encode('utf-8','ignore').decode("utf-8") for x in tags_processed['word']]

# Count occurences
tags_processed_lowercase = [str(word).lower() for word in tags_processed['word']]
word_freq = Counter(tags_processed_lowercase)
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())

common_words_out.to_csv(PATH_OUT,sep=';')
display(common_words_out.head(30))


# %%

######### Comparison of communities

            ######################
            ######################
            ###### ANALYSIS ######
            ######################
            ######################


# Filtering

result = []

#['fake','maga','security','trump','wall','shutdown','economic','carbon','climate','brexit','eu','china']
targets_list =['harassed','satanic','pedophile','shutdown']

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
                temp.loc[len(temp.index)] = [i,'-', '-', 0]
        else:
            if (titles['word']==target).any():
                temp.loc[len(temp.index)] = [i, titles[titles['word']==target].index.values[0], '-', float(titles[titles['word']==target]['frequency'])]
            else:
                display(target + " is not in community " + str(i))
                temp.loc[len(temp.index)] = [i,'-', '-', 0]

    result = result + [temp]

    display(target)
    display(temp)


# %%


            ######################
            ######################
            ####### GRAPHS #######
            ######################
            ######################


# %%

############ GRAPH : Words occurences

data = pd.DataFrame()
for i in range(0,len(targets_list)):
    data.insert(i, column=targets_list[i], value=result[i]['title_frequency'])

display(data)

fig = px.line(data, title='Topics frequencies in different communities',labels={'value':'frequency','index':'community','variable':'Topics :'},width=600, height=400)
not_active_traces = [targets_list[i] for i in range(len(targets_list)) if i>len(targets_list)*1/3]
fig.for_each_trace(lambda trace: trace.update(visible='legendonly') if trace.name in not_active_traces else ())
fig.show()

#plt.show(sns.lineplot(data=data))

# %%

############ GRAPH : Closeness matrix

closeness_matrix = pd.DataFrame(index=communities,columns=communities)
for x in communities:
    for y in communities:
        for target in range(len(targets_list)):
            a = int(result[target][result[target]['community']==x]['rank_title'])
            b = int(result[target][result[target]['community']==y]['rank_title'])
            c = 0
            d = 0
            if use_tags:
                c = int(result[target][result[target]['community']==x]['rank_tag'])
                d = int(result[target][result[target]['community']==y]['rank_tag'])
            closeness_matrix[x][y]=abs(a-b)+abs(c-d)

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

# %%

############ GRAPH : 20 first occurences in titles

sns.set(font_scale=1.5)
fig, ax = plt.subplots(2, 3, figsize=(45, 30))

palette = sns.color_palette("coolwarm", n_colors = 100)
palette.reverse()

for community in communities:
    position_y = int(community/3)
    position_x = community % 3
    
    data=pd.read_csv(DIR_OUT+"communities_comparison/titles_occurences_"+str(community)+"_"+word+"_"+start_date+"_"+end_date+".csv", sep=';')
    data=data.head(30).sort_values(by='occurences', ascending=False)
    sns.barplot(x = 'frequency', y = 'word', data=data, ax = ax[position_y,position_x], palette=palette)
    ax[position_y,position_x].set_title("Most cited topics in community "+str(community), fontsize = 25)
    ax[position_y,position_x].set_xlabel("Frequency", fontsize = 20)
    ax[position_y,position_x].set_ylabel("Topics", fontsize = 20)


# %%


# Permettre de choisir les mots que l'on veut avoir dans la matrice de 'closeness' des chaînes
# Plan idée : graph, graph sans la merde, orientation politique des utilisateurs, plus loin que l'orientation politique : des perspectives différentes sur l'actualité (occurences de mots)
# Idée distance map, avec des axes comme droite/gauche, international ou pas, conspi ou pas, regard sur Trump
# Raconter ce qu'il s'est passé dans cette période de fin 2018 début 2019
# Une couleur associée à chaque communauté pour tout le long



# %%

######## Creating a smaller copy of the LOCO.json file

# chunk = 10 000 000   ->  9 766 KB
# total file : chunk = 587 622 363

text = ""
chunk = 100000000

with codecs.open("data/LOCO.json", 'r', encoding='latin1') as file:
    #writing the header
    text = file.read(chunk)
file.close()

with codecs.open("data/LOCO_small.txt", 'w', encoding='latin1') as file:
    file.write(text)
file.close()

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
    doc_processed = [token.replace("'",'') for token in doc_processed]
    doc_processed = [token.replace('"','') for token in doc_processed]
    doc_processed = [token.replace("\\",'') for token in doc_processed]
    doc_processed = [token.replace("/",'') for token in doc_processed]
    doc_processed = [token.replace("\n",'') for token in doc_processed]
    doc_processed = [token.replace("â",'') for token in doc_processed]

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

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())
common_words_out = common_words_out.sort_values(by='frequency', ascending = False)[['word','frequency']]

common_words_out.to_csv(DIR_OUT+"loco_occurences.csv")

display(common_words_out.head(10))

# %% 

###### Comparing word occurences in LOCO and in our titles for a given

loco_occurences = pd.read_csv(DIR_OUT+"loco_occurences.csv", index_col = 0)
distance = []

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

    distance = distance + [math.sqrt(merged['squared_distance'].sum())]

display(distance)

fig = px.line(distance, title='Distance between topics in a community and topics in conspiracist medias',labels={'index':'community','value':'distance'},width=650, height=400)
fig.update_traces(showlegend=False)
fig.show()


# %%

## Data for the regression
df = pd.DataFrame({'y':result['leftness'], 'rank_ti':result['rank_title'], 'rank_ta':result['rank_tag']})

## Perform regression using statsmodels ols function
model = smf.ols('y ~ rank_ti + rank_ta', data = df)
results = model.fit()
#print(results.summary())


