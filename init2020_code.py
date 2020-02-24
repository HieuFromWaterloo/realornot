# %% [markdown]
# <a class="anchor" id="0"></a>
# # Real or not

# %% [markdown]
# # Acknowledgements
#
# This kernel uses such good kernels:
# * https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert
# * https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
# * https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
# * https://www.kaggle.com/itratrahman/nlp-tutorial-using-python
# * https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
# * https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic
# * https://www.kaggle.com/khoongweihao/bert-base-tf2-0-minimalistic-iii
# * https://www.kaggle.com/vbmokin/disaster-nlp-keras-bert-using-tfhub-tuning
# * https://www.kaggle.com/user123454321/bert-starter-inference
# * https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# * https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data
# * https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
# * https://tfhub.dev/s?q=bert

# %% [markdown]
# <a class="anchor" id="0.1"></a>
# ## Table of Contents
#
# 1. [My upgrade BERT model](#1)
#     -  [Commit now](#1.1)
#     -  [Previous commits: Dropout = 0.1 or 0.3](#1.2)
#     -  [Previous commits: epochs = 3](#1.3)
#     -  [Previous commits: epochs = 4](#1.4)
#     -  [Previous commits: epochs = 5](#1.5)
#     -  [Previous commits: with training tweets correction](#1.6)
#     -  [Previous commits: parameters and LB scores](#1.7)
# 1. [Import libraries](#2)
# 1. [Download data](#3)
# 1. [EDA](#4)
# 1. [Data Cleaning](#5)
# 1. [WordCloud](#6)
# 1. [Bag of Words Counts](#7)
# 1. [TF IDF](#8)
# 1. [GloVe](#9)
# 1. [BERT using TFHub](#10)
#    - [Submission by BERT](#10.1)
# 1. [Showing Confusion Matrices](#11)

# %% [markdown]
# ## 2. Import libraries <a class="anchor" id="2"></a>
#
# [Back to Table of Contents](#0.1)

# %% [code]
import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from nltk.corpus import stopwords
from nltk.util import ngrams

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report,confusion_matrix

from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))

import re
from nltk.tokenize import word_tokenize
import gensim
import string

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam

# %% [markdown]
# ## 3. Download data <a class="anchor" id="3"></a>
#
# [Back to Table of Contents](#0.1)

# %% [code]
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')
test= pd.read_csv('../input/nlp-getting-started/test.csv')
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

# %% [code]
# # From https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data -
# # author of this kernel read tweets in training data and figure out that some of them have errors:
# ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
# tweet.loc[tweet['id'].isin(ids_with_target_error),'target'] = 0
# tweet[tweet['id'].isin(ids_with_target_error)]

# %% [code]
print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))
print('There are {} rows and {} columns in train'.format(test.shape[0],test.shape[1]))

# %% [code]
tweet.head(10)

# %% [markdown]
# ## 4. EDA <a class="anchor" id="4"></a>
#
# [Back to Table of Contents](#0.1)

# %% [markdown]
# Thanks to:
# * https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
# * https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
# * https://www.kaggle.com/itratrahman/nlp-tutorial-using-python

# %% [markdown]
# ### Class distribution

# %% [markdown]
# Before we begin with anything else, let's check the class distribution.

# %% [code]
# extracting the number of examples of each class
Real_len = tweet[tweet['target'] == 1].shape[0]
Not_len = tweet[tweet['target'] == 0].shape[0]
print(Real_len)
print(Not_len)

# %% [code]
# bar plot of the 3 classes
plt.rcParams['figure.figsize'] = (7, 5)
plt.bar(10,Real_len,3, label="Real", color='blue')
plt.bar(15,Not_len,3, label="Not", color='red')
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propertion of examples')
plt.show()

# %% [markdown]
# ### Number of characters in tweets

# %% [code]
tweet['length'] = tweet['text'].apply(lambda x: len(x))

# %% [code]
plt.rcParams['figure.figsize'] = (13, 7)
plt.hist(tweet[tweet['target'] == 0]['length'], alpha = 0.6, bins=150, label='Not')
plt.hist(tweet[tweet['target'] == 1]['length'], alpha = 0.8, bins=150, label='Real')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,150)
# plt.grid()
plt.show()

# %% [code]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,7))
tweet_len=tweet[tweet['target']==1]['text'].str.len()
ax1.hist(tweet_len,color='blue',alpha = 0.6)
ax1.set_title('disaster tweets')
tweet_len=tweet[tweet['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='red',alpha = 0.6)
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()

# %% [markdown]
# The distribution of both seems to be almost same.120 t0 140 characters in a tweet are the most common among both.

# %% [markdown]
# ### Number of words in a tweet

# %% [code]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=tweet[tweet['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='blue', alpha=0.6)
ax1.set_title('disaster tweets')
tweet_len=tweet[tweet['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='red', alpha=0.6)
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.show()


# %% [markdown]
# ###  Average word length in a tweet

# %% [code]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
word=tweet[tweet['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='blue')
ax1.set_title('disaster')
word=tweet[tweet['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('Not disaster')
fig.suptitle('Average word length in each tweet')

# %% [code]
def create_corpus(target):
    corpus=[]

    for x in tweet[tweet['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

# %% [code]
def create_corpus_df(tweet, target):
    corpus=[]

    for x in tweet[tweet['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

# %% [markdown]
# ### Common stopwords in tweets

# %% [markdown]
# First we  will analyze tweets with class 0.

# %% [code]
corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]

# %% [code]
# displaying the stopwords
np.array(stop)

# %% [code]
plt.rcParams['figure.figsize'] = (13, 7)
x,y=zip(*top)
plt.bar(x,y, alpha = 0.6)

# %% [markdown]
# Now,we will analyze tweets with class 1.

# %% [code]
corpus=create_corpus(1)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]


plt.rcParams['figure.figsize'] = (13, 7)
x,y=zip(*top)
plt.bar(x,y, alpha=0.6)

# %% [markdown]
# In both of them,"the" dominates which is followed by "a" in class 0 and "in" in class 1.

# %% [markdown]
# ### Analyzing punctuations

# %% [markdown]
# First let's check tweets indicating real disaster.

# %% [code]
plt.figure(figsize=(13,7))
corpus=create_corpus(1)

dic=defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1

x,y=zip(*dic.items())
plt.bar(x,y)

# %% [markdown]
# Now,we will move on to class 0.

# %% [code]
plt.figure(figsize=(13,7))
corpus=create_corpus(0)
dic=defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1

x,y=zip(*dic.items())
plt.bar(x,y,color='green')

# %% [markdown]
# ### Common words

# %% [code]
plt.figure(figsize=(13,7))
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:40]:
    if (word not in stop) :
        x.append(word)
        y.append(count)
sns.barplot(x=y,y=x)

# %% [markdown]
# Lot of cleaning needed !

# %% [markdown]
# ### N-gram analysis

# %% [markdown]
# we will do a bigram (n=2) analysis over the tweets. Let's check the most common bigrams in tweets.

# %% [code]
def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(1, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# %% [code]
plt.figure(figsize=(16,5))
top_tweet_bigrams=get_top_tweet_bigrams(tweet['text'])[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x, alpha=0.6)

# %% [markdown]
# ## 5. Data Cleaning <a class="anchor" id="5"></a>
#
# [Back to Table of Contents](#0.1)

# %% [markdown]
# Thanks to https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

# %% [code]
df=pd.concat([tweet,test])
df.shape

# %% [markdown]
# ### Removing urls

# %% [code]
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"

# %% [code]
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

remove_URL(example)

# %% [code]
df['text']=df['text'].apply(lambda x : remove_URL(x))

# %% [markdown]
# ### Removing HTML tags

# %% [code]
example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""

# %% [code]
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))

# %% [code]
df['text']=df['text'].apply(lambda x : remove_html(x))

# %% [markdown]
# ### Removing Emojis

# %% [code]
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

remove_emoji("Omg another Earthquake 😔😔")

# %% [code]
df['text']=df['text'].apply(lambda x: remove_emoji(x))

# %% [markdown]
# ### REmove non ASCII and links

# %% [code]
def clean_tweets(tweet):
    """Removes links and non-ASCII characters"""

    tweet = ''.join([x for x in tweet if x in string.printable])

    # Removing URLs
    tweet = re.sub(r"http\S+", "", tweet)

    return tweet

# %% [markdown]
# ### Removing punctuations

# %% [code]
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="I am a #king"
print(remove_punct(example))

# %% [code]
df['text']=df['text'].apply(lambda x : remove_punct(x))

# %% [code]
df['text']=df['text'].apply(lambda x : x.lower())

# %% [code]
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

# %% [markdown]
# ### Stemming & Lemmatizing

# %% [code]
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

#create an object of class PorterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()

# Some examples from datacamps: https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
#proide a word to be stemmed
print("Porter Stemmer")
print(porter.stem("cats"))
print(porter.stem("trouble"))
print(porter.stem("troubling"))
print(porter.stem("troubled"))
print("\n")
print("Lancaster Stemmer")
print(lancaster.stem("cats"))
print(lancaster.stem("trouble"))

# %% [code]
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))

# %% [code]
def lemmatization(text):
    tmp = text.split()
    tmp = [wordnet_lemmatizer.lemmatize(i, "a") for i in tmp]
    return " ".join(tmp)

example="He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
print(lemmatization(example.lower()))

# %% [code]
df['text'].head(10)

# %% [code]
df['text'].head(10).apply(lambda x : lemmatization(x))

# %% [code]
 df['text'].apply(lambda x: re.sub(r'[^\w]', ' ', x))

# %% [code]
df['text'] = df['text'].apply(lambda x : lemmatization(x))

# %% [markdown]
# ## 7. Bag of Words Counts <a class="anchor" id="7"></a>
#
# [Back to Table of Contents](#0.1)

# %% [markdown]
# Thanks to https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb

# %% [code]
def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

list_corpus = df["text"].tolist()
list_labels = df["target"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                                                random_state=40)

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# %% [markdown]
# ### Visualizing the embeddings

# %% [code]
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.6, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='orange', label='Not')
            blue_patch = mpatches.Patch(color='blue', label='Real')
            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 15})

fig = plt.figure(figsize=(13, 7))
plot_LSA(X_train_counts, y_train)
plt.show()

# %% [markdown]
# These embeddings don't look very cleanly separated. Let's see if we can still fit a useful model on them.

# %% [markdown]
# ## 8. TF IDF <a class="anchor" id="8"></a>
#
# [Back to Table of Contents](#0.1)

# %% [code]
df.dropna(subset=["target"], inplace=True)

# %% [code]
columns = df.columns
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': columns,
                                 'percent_missing (%)': percent_missing})
missing_value_df.sort_values('percent_missing (%)', inplace=True, ascending=False)
missing_value_df.head(50)

# %% [code]
def tfidf(data):
    # tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(2, 2), stop_words='english')

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# %% [code]
fig = plt.figure(figsize=(13, 7))
plot_LSA(X_train_tfidf, y_train)
plt.show()

# %% [code]
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')


# list_corpus = df["text"]#.tolist()

features = tfidf.fit_transform(df["text"]).toarray()
labels = df["target"]#.tolist()
features.shape

# %% [markdown]
# > eaach of 7613 headlines is represented by 3388 features, representing the tf-idf score for different unigrams and bigrams.

# %% [code]
# category_id_df = df[['label', 'target']].drop_duplicates().sort_values('category_id')
# category_to_id = dict(category_id_df.values)

# for label, category_id in sorted(category_to_id.items()):
#     print(label, category_id)
category_to_id = {"fake":0, "real":1}
category_to_id

# %% [code]
from sklearn.feature_selection import chi2

#  find the terms that are the most correlated with each of the categories:
N = 2
for label, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    trigrams = [v for v in feature_names if len(v.split(' ')) == 3]

    print("# '{}':".format(label))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# %% [code]
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.datasets import make_classification
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.calibration import CalibratedClassifierCV
import string
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.stem import SnowballStemmer,PorterStemmer
from nltk.corpus import stopwords,state_union
import os
import random
from statistics import mode
from io import StringIO
import re
import pickle
import sys, getopt

# %% [code]
%%time
models = [
    RandomForestClassifier(),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(),
    SGDClassifier(),
    SVC()
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

# %% [code]
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=7, jitter=True, edgecolor="gray", linewidth=2)

sns.set(rc={'figure.figsize':(12, 12)})
plt.show()

# %% [code]
cv_df.groupby('model_name').accuracy.mean().sort_values(ascending=False)

# %% [code]
%%time
import pickle
model = MultinomialNB()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=42,
                                                                                shuffle=True)

# svm = LinearSVC()
# model = CalibratedClassifierCV(svm)
# clf.fit(X_train, y_train)
# y_proba = clf.predict_proba(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

# with open("pickled_algos/August1st_TFIDF_LinearSVC.pickle","wb") as f:
#     pickle.dump(model, f)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('\nClasification report:\n', classification_report(y_test, y_pred))


print("\nConfusion Matrix is very confusing")
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap = "Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# %% [markdown]
# Testing accuracy 0.7944845699277742
#
# Testing F1 score: 0.7894329084574213
#
# Clasification report:
#                precision    recall  f1-score   support
#
#          0.0       0.78      0.90      0.83       874
#          1.0       0.83      0.65      0.73       649
#
#     accuracy                           0.79      1523
#    macro avg       0.80      0.78      0.78      1523
# weighted avg       0.80      0.79      0.79      1523

# %% [code]
test_pred_tfidf = model_BERT.predict(test_input)
test_pred_tfidf_int = test_pred_tfidf.round().astype('int')
submission['target'] = test_pred_BERT_int
submission.to_csv("submission_bert_nodropout30.csv", index=False, header=True)

# %% [markdown]
# > ## 9. GloVe <a class="anchor" id="9"></a>
#
# [Back to Table of Contents](#0.1)

# %% [markdown]
# Thanks to https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

# %% [markdown]
# Here we will use GloVe pretrained corpus model to represent our words. It is available in 3 varieties : 50D, 100D and 200 Dimentional. We will try 100D here.

# %% [code]
def create_corpus_new(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus

# %% [code]
corpus=create_corpus_new(df)

# %% [code]
embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

# %% [code]
MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

# %% [code]
word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))

# %% [code]
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec

# %% [code]
tweet_pad[0][0:]

# %% [markdown]
# ## Baseline Model with GloVe results

# %% [code]
model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.25))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=3e-4)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

# %% [code]
model.summary()

# %% [code]
train=tweet_pad[:tweet.shape[0]]
test=tweet_pad[tweet.shape[0]:]

# %% [code]
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.2)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)

# %% [code]
fig = plt.figure(figsize=(13, 7))
plot_LSA(train,tweet['target'])
plt.show()

# %% [code]
%%time
# 10-20 epochs
history=model.fit(X_train,y_train,batch_size=4,epochs=13,validation_data=(X_test,y_test),verbose=2)

# %% [markdown]
# Epoch 13/13
#  - 84s - loss: 0.4711 - accuracy: 0.7877 - val_loss: 0.4470 - val_accuracy: 0.7958
#
# CPU times: user 23min 17s, sys: 1min 10s, total: 24min 28s
#
# Wall time: 17min 5s

# %% [code]
train_pred_GloVe = model.predict(train)
train_pred_GloVe_int = train_pred_GloVe.round().astype('int')
test_pred_GloVe = model.predict(test)
test_pred_GloVe_int = test_pred_GloVe.round().astype('int')
submission['target'] = test_pred_GloVe_int
submission.to_csv("submission_glove.csv", index=False, header=True)
submission.head(10)

# %% [code]
pred = pd.DataFrame(train_pred_GloVe, columns=['preds'])
pred.plot.hist()

# %% [markdown]
# ## 10. BERT using TFHub <a class="anchor" id="10"></a>
#
# [Back to Table of Contents](#0.1)

# %% [markdown]
# https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# %% [code]
# We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

# %% [code]
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow_hub as hub

import tokenization

# %% [code]
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# %% [code]
Dropout_num = 0
learning_rate = 3e-6
valid = 0.2
epochs_num = 5
batch_size_num = 8
target_corrected = True
target_big_corrected = True

# %% [code]
def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]

    if Dropout_num == 0:
        # Without Dropout
        out = Dense(1, activation='sigmoid')(clf_output)
    else:
        # With Dropout(Dropout_num), Dropout_num > 0
        x = Dropout(Dropout_num)(clf_output)
        out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# %% [code]
# Load BERT from the Tensorflow Hub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

# %% [code]
# Load CSV files containing training data
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# %% [code]
# Load tokenizer from the bert layer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# %% [code]
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

def clean(df):
    df['text'] = df['text'].apply(lambda x : remove_URL(x))
    df['text'] = df['text'].apply(lambda x : remove_html(x))
    df['text'] = df['text'].apply(lambda x: remove_emoji(x))
    df['text'] = df['text'].apply(lambda x : remove_punct(x))
    df['text'] = df['text'].apply(lambda x : convert_abbrev_in_text(x))
    df['text'] = df['text'].apply(lambda x : clean_tweets(x))
    df['text'] = df['text'].apply(lambda x : lemmatization(x))
    return df

# Thanks to https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data -
# author of this kernel read tweets in training data and figure out that some of them have errors:
if target_corrected:
    ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
    train.loc[train['id'].isin(ids_with_target_error),'target'] = 0
    train[train['id'].isin(ids_with_target_error)]


if target_big_corrected:
    train = clean(train)
    test = clean(test)

# %% [code]
# Encode the text into tokens, masks, and segment flags
train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values

# %% [markdown]
# ---

# %% [markdown]
# ## NO DROPOUT

# %% [code]
# Model: Build, Train, Predict, Submit -- DROPOUT
Dropout_num = 0.3

model_BERT = build_model(bert_layer, max_len=160)
model_BERT.summary()

# %% [code]
%%time
# DOROPOUT:
checkpoint = ModelCheckpoint('model_dropout.h5', monitor='val_loss', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

train_history = model_BERT.fit(
    train_input, train_labels,
    validation_split = valid,
    epochs = epochs_num,
    callbacks=[checkpoint, es],
    batch_size = batch_size_num
)

# %% [markdown]
# Epoch 2/4
#
# 6090/6090 [==============================] - 127s 21ms/sample - loss: 0.2765 - accuracy: 0.8966 - val_loss: 0.4238 - val_accuracy: 0.8181
#
# Epoch 00002: early stopping

# %% [code]
test_pred_BERT = model_BERT.predict(test_input)
test_pred_BERT_int = test_pred_BERT.round().astype('int')

submission['target'] = test_pred_BERT_int
submission.to_csv("submission_bert_dropout30_8332_5e-5.csv", index=False, header=True)

# %% [markdown]
# ---

# %% [markdown]
# ## DROPOUT

# %% [code]
# Model: Build, Train, Predict, Submit -- NO DROPOUT
model_BERT = build_model(bert_layer, max_len=160)
model_BERT.summary()

# %% [code]
%%time
# NO DROPOUT
checkpoint = ModelCheckpoint('model_dropout.h5', monitor='val_loss', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

train_history = model_BERT.fit(
    train_input, train_labels,
    validation_split = valid,
    epochs = epochs_num,
    callbacks=[checkpoint, es],
    batch_size = batch_size_num
)

# %% [code]
train_pred_BERT = model_BERT.predict(train_input)
train_pred_BERT_int = train_pred_BERT.round().astype('int')
test_pred_BERT = model_BERT.predict(test_input)
test_pred_BERT_int = test_pred_BERT.round().astype('int')
submission['target'] = test_pred_BERT_inta
submission.to_csv("submission_bert_nodropout.csv", index=False, header=True)

# %% [markdown]
# [Go to Top](#0)
