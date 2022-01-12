import spacy
import json
import multiprocessing # allows to use multiple cores to work
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
import numpy as np
import pandas as pd
from time import time
import re
import pickle
from collections import defaultdict

# To train gensim we need data training dispose in the next forms:
# "Tom was a cat. Jerry was a mouse" [['Tom','was','cat'], ['Jerry','was','mouse']]
# And a vector of stopwords: Which are those words who appear so much times that 
# decrease Machine Learning performance. 
# Words like 'a','an','the','has','have', etc.['a','an','the','has','have']

# Create data in form of WordVectors using gensim based on this tutorial:
# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial/notebook
# it is also in ./24-01-sub-tutorial-creating-word-vector-data-training

with open('./sources/harry_potter.txt','r',encoding='utf-8') as f:
    text = f.read()
    chapters = text.split('CHAPTER ')[:]
    segments = np.array([])
    for chapter in chapters:
        segments = np.append(segments,chapter.split('\n\n')[2:])  # Only text without the title

# # Cleaning
# # We are lemmatizing and removing the stopwords and non-alphabetic characters 
# # for each line of dialogue.
nlp = spacy.load('en_core_web_sm', disable=['ner','parser']) # Disabling Named Entity Recognition for speed

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

# Removes non-alphabetic characters:
# We create a function called brief_cleaning (generator class) which call
# allow you to declare a function that behaves like an iterator, 
# i.e. it can be used in a for loop just once. This function iterate over the dialogs, 
# converts it to lowercase and replace non-alphabetic characters with spaces ' '
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in segments)

# # # Taking advantage of spaCy .pipe() attribute to speed-up the cleaning process:
t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

# Put the results in a DataFrame to remove missing values and duplicates:
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
print(df_clean.head())

# Bigrams: We are using Gensim Phrases package to automatically detect common
# phrases (bigrams) from a list of sentences.
# https://radimrehurek.com/gensim/models/phrases.html
# The main reason we do this is to catch words like "mr_burns" or "bart_simpson" !
# As Phrases() takes a list of list of words as input:
sent = [row.split() for row in df_clean['clean']]
print(sent[:2])
# Creates the relevant phrases from the list of sentences:
phrases = Phrases(sent, min_count=30, progress_per=10000)
# # The goal of Phraser() is to cut down memory consumption of Phrases(), by
# # discarding model state not strictly needed for the bigram detection task:
bigram = Phraser(phrases)
# # Transform the corpus based on the bigrams detected:
sentences = bigram[sent]

# transform "sentences" to a dict. We could use in "sentences form" but
# in this tutorial it's a json file.
final_data = []
for sent in sentences:
    final_data.append(sent)

# Save data accord to the tutorial into a json file called 'hp.json'
with open('./sources/hp.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=4)

