# %%

import pandas as pd
import spacy
import codecs
import csv
import matplotlib.pyplot as plt

# Word occurences count
from collections import Counter

# Sentiment analysis
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DIR = "data/"
DIR_LARGE = "data/large/"
DIR_OUT = "csv_outputs/"
PATH_METADATA = DIR_LARGE + "yt_metadata_en.jsonl.gz"

# %%

############# Create a list of all videos of a selected communities, with titles and tags : use display_id_to_channels_title_tags

selected_commu = '2'

channels_list = pd.read_csv("csv_outputs/louvain_communities.csv", sep=";", usecols=[selected_commu])
channels_list.columns=['channel_num']
display(channels_list.head())

pd.DataFrame(columns=['title','upload_date']).to_csv(DIR_OUT+"titles_date.csv", sep=';', index=False, header=True)
pd.DataFrame(columns=['tags','upload_date']).to_csv(DIR_OUT+"tags_date.csv", sep=';', index=False, header=True)

for chunk in pd.read_csv(DIR_OUT + "channel_num_to_title_tags_date.csv", sep=";", chunksize=100000):
    chunk = channels_list.merge(chunk, on='channel_num')
    titles = pd.DataFrame(chunk[['title','upload_date']])
    titles.to_csv(DIR_OUT+"titles_date.csv", sep=';', mode='a', index=False, header=False)
    tags = pd.DataFrame(chunk[['tags','upload_date']])
    tags.to_csv(DIR_OUT+"tags_date.csv", sep=';', mode='a', index=False, header=False)

# EXEC : ~1min

# %%

############# Data used for processing of titles and tags

nlp = spacy.load('en_core_web_sm')
nlp.max_length=1000000

# Filtering
word = ''
period_start = pd.to_datetime("2013-02-06", format='%Y-%m-%d')
period_end =pd.to_datetime("2022-02-06", format='%Y-%m-%d')

# Processing
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
#news_lexical_field = ['news','News','wrap','Day','day','TV','Channel','channel']
news_lexical_field = []
my_undesired_list = ['|','l',word] + news_lexical_field

# %%

            ######################
            ######################
            ####### TITLES #######
            ######################
            ######################

############# Select videos whom title contains a given word

titles = pd.read_csv(DIR_OUT + "titles_date.csv",sep=";")

titles_contain = titles[titles['title'].str.contains(word, case=False)]
titles_contain.to_csv(DIR_OUT + "titles_date_f.csv",sep=";",index=False, header=True)
#titles_contain.to_csv(DIR_OUT+"titles.txt", sep="\n",index=False, header=False)

display(titles_contain)

# %%

############# Select videos in a time period of interest

titles = pd.read_csv(DIR_OUT + "titles_date_f.csv",sep=";")
titles["upload_date"] = pd.to_datetime(titles["upload_date"], format='%Y-%m-%d %H:%M:%S')
titles = titles[(titles['upload_date']>=period_start) & (titles['upload_date']<=period_end)]
#display(titles)

titles['title'] = [x.encode('utf-8','ignore').decode("utf-8") for x in titles['title']]
titles["title"].to_csv(DIR_OUT+"titles_to_process.txt", sep="\n",index=False, header=False)

# %%

############# Process the titles 

books = list()

with codecs.open(os.path.join(DIR_OUT,"titles_to_process.txt")) as f:
    books.append(f.read())

# Remove new lines
books = [" ".join(b.split()) for b in books]
#display("Length of the book is " + str(len(books[0])/nlp.max_length) + " times the max length that can be processed at once.")

# Remove stopwords and punctuation
for i in range(0,int(len(books[0])/nlp.max_length)+1):
    #display('iteration ' + str(i))
    doc = nlp(books[0][i*nlp.max_length:(i+1)*nlp.max_length])

    doc_processed = [token.text for token in doc if (not token.is_digit and not token.is_stop)]
    doc_processed = [token.text for token in doc if (not token.is_punct and not (token.text in my_undesired_list))]
    doc_processed = [token.replace(',','') for token in doc_processed]

    with open(DIR_OUT+"titles_words.csv", 'w') as file:
        #writing the header
        file.write('word\n')
        # writing the data
        for x in doc_processed:
            file.write(x+'\n')

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

titles_processed = pd.read_csv(DIR_OUT+"titles_words.csv", sep=',')
#titles_processed = [x.encode('utf-8','ignore').decode("utf-8") for x in titles_processed['word']]

# Count occurences
#titles_processed_lowercase = [word.lower() for word in titles_processed]
word_freq = Counter(titles_processed['word'])
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

display(common_words)

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())

common_words_out.to_csv(DIR_OUT+"titles_occurences.csv",sep=';')
display(common_words_out.head(30))


# %%

            ######################
            ######################
            ####### TAGS #########
            ######################
            ######################


############# Select videos whom tag contains a given word

tags = pd.read_csv(DIR_OUT + "tags_date.csv",sep=";")

tags = tags[~(tags['tags'].isnull())]

tags_contain = tags[tags['tags'].str.contains(word, case=False)]
tags_contain.to_csv(DIR_OUT + "tags_date_f.csv",sep=";",index=False, header=True)
#tags_contain.to_csv(DIR_OUT+"tags.txt", sep="\n",index=False, header=False)

display(tags_contain)

# %%

############# Select videos in a time period of interest

period_start = pd.to_datetime("2013-02-06", format='%Y-%m-%d')
period_end =pd.to_datetime("2022-02-06", format='%Y-%m-%d')

tags = pd.read_csv(DIR_OUT + "tags_date_f.csv",sep=";")
tags["upload_date"] = pd.to_datetime(tags["upload_date"], format='%Y-%m-%d %H:%M:%S')
tags = tags[(tags['upload_date']>=period_start) & (tags['upload_date']<=period_end)]
display(tags)

tags["tags"].to_csv(DIR_OUT+"tags_to_process.txt", sep="\n",index=False, header=False)

# %%

############# Process the tags 

books = list()

with codecs.open(os.path.join(DIR_OUT,"tags_to_process.txt")) as f:
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

tags_processed = pd.read_csv(DIR_OUT+"tags_words.csv", sep=',')
#tags_processed = [x.encode('utf-8','ignore').decode("utf-8") for x in tags_processed['word']]

# Count occurences
#tags_processed_lowercase = [word.lower() for word in tags_processed]
word_freq = Counter(tags_processed)
common_words = word_freq.most_common()
common_words_out = pd.DataFrame(common_words)
common_words_out.columns=['word','occurences']

common_words_out.insert(2, column='frequency', value=common_words_out['occurences']/common_words_out['occurences'].sum())

common_words_out.to_csv(DIR_OUT+"tags_occurences.csv",sep=';')
display(common_words_out.head(30))

# %%

