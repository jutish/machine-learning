import pandas as pd
import numpy as np
import spacy
import praw
import os
import re
from xgboost import XGBRegressor
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from time import time
import json
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing


### Create training data for train our custom WordVector ###

# Load data
ruddit_df = pd.read_csv('./data/ruddit_comments_score.csv')
# print(ruddit_df.head())
validt_df = pd.read_csv('./data/validation_data.csv')
# print(validt_df.head())
comm2s_df = pd.read_csv('./data/comments_to_score.csv')
# print(comm2s_df.head())

## Merge all data in one DataFrame ##
# Flat validation data into a one single column
validt_df = pd.DataFrame({'comment': 
    validt_df[['less_toxic','more_toxic']].values.flatten()})
# Get a pandas Series with all the comments
data_df = ruddit_df['body']
data_df = data_df.append(validt_df['comment'])
data_df = data_df.append(comm2s_df['text'])
data_df.index.name = 'comment_id'


## Clean data
def remove_deleted_comments(data):
    print('Before remove [deleted] comments: ', data.shape)
    data = data.loc[data != '[deleted]'] # .reset_index(drop=True)
    print('After remove [deleted] comments: ', data.shape)
    return data

def remove_blank_comments(data):
    print('Before remove Blanks comments: ', data.shape)
    data = data.loc[data != ' '] #.reset_index(drop=True)
    print('After remove Blanks comments: ', data.shape)
    return data

def remove_nan_comments(data):
    print('Before remove NaN comments: ', data.shape)
    data = data.dropna()#.reset_index(drop=True)
    print('After remove NaN comments: ', data.shape)
    return data


def remove_duplicate_comments(data):
    print('Before remove duplicate comments: ', data.shape)
    data = data.drop_duplicates() #reset_index(drop=True)
    print('After remove duplicate comments: ', data.shape)
    return data


def remove_punct_and_spaces(text):
    # removing links
    temp_string = re.sub("http[s]?://(?:[a-zA-Z]|[0–9]|[$-_@.&+]|(?:%[0–9a-fA-F]\
        [0–9a-fA-F]))+", ' ', text)
    # removing all everything except a-z english letters
    regex = re.compile('[^a-zA-Z]')
    temp_string = regex.sub(' ', temp_string)
    # removing extra spaces
    text = re.sub(' +', ' ', temp_string).lower()
    return text


def remove_stop_words(text, stop_words):
    big_regex =  re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stop_words)))
    text = big_regex.sub("", text)
    return text


def remove_stop_words_and_punct(data):
    nlp = spacy.load('en_core_web_sm')
    stop_words = list(nlp.Defaults.stop_words)
    data = data.apply(remove_stop_words, stop_words=stop_words)
    data = data.apply(remove_punct_and_spaces)
    return data


def get_lemma_text(doc, model):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    doc = model(doc)
    txt = [token.lemma_ for token in doc]
    if len(txt) > 2:
        return' '.join(txt)


def lemmatize_comments(data):
    nlp = spacy.load('en_core_web_sm')
    data = data.apply(get_lemma_text, model=nlp)
    return data


def lower_case_comments(data):
    data = pd.Series([comment.lower() for comment in data], index=data.index)
    return data


def remove_unicode(data):
    pattern = r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"
    data = pd.Series([re.sub(pattern,"",comment) for comment in data], index=data.index)
    return data


def remove_stop_words_v2(data):
    nlp = spacy.load('en_core_web_sm')
    stop = list(nlp.Defaults.stop_words)
    data = pd.Series([" ".join([word for word in text.split() if word not in (stop)])
        for text in data], index=data.index)
    return data


def lemmatize_comments_v2(data):
    nlp = spacy.load('en_core_web_lg')
    data = pd.Series([" ".join([token.lemma_ for token in nlp(text)])
        for text in data], index=data.index)
    return data


def clean_data(data):
    t = time()
    data = remove_blank_comments(data)
    data = remove_deleted_comments(data)
    data = remove_nan_comments(data)
    data = remove_duplicate_comments(data)
    data = lower_case_comments(data)
    data = remove_unicode(data)  # Punct and http adresses, emojis and more.
    data = remove_stop_words_v2(data)
    data = lemmatize_comments_v2(data)
    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    return data


def split_data(data):
    if data != None and len(data) > 0:
        return data.split()
    else:
        print('none find')
        return []


def get_bigram(data, phraser):
    return phraser[data]


# Transform each sentence into [[word1,word2,..,wordN], [word1,...wordN]]
# if its posible get bigrams of words using Gensim Phraser
def transform_data(data):
    data = data.apply(split_data)
    phrases = Phrases(data, min_count=3, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
    phraser = Phraser(phrases) # The goal of Phraser() is to cut down memory consumption of Phrases()
    data = data.apply(get_bigram, phraser=phraser)
    return data


def create_training_data(data, path):
    data = clean_data(data)
    data = transform_data(data)
    data = [sentence for sentence in data]
    # Save data accord to the tutorial into a json file called --> path
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# Create training data, and save it.
path = './data/training_data_for_custom_vector.json'
data = data_df
# create_training_data(data, path)

# ############################################
# ## Training our word2vec model            ##
# ############################################

def training(model_name, data_path):
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2,
        window=2,
        vector_size = 300,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores-1)
    w2v_model.build_vocab(data)
    w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=30)
    w2v_model.save(f'data/{model_name}.model') # save the model
    w2v_model.wv.save_word2vec_format(f'data/word2vec_{model_name}.txt')

path = './data/training_data_for_custom_vector.json'
# training('toxicity_model', path)

# ############################################
# ## Load the Custom Word Vector into Spacy ##
# ############################################

import subprocess
import sys

model_name = './data/toxicity_custom_model'
word_vectors = './data/word2vec_toxicity_model.txt'

# Usage: python -m spacy init vectors [OPTIONS] LANG VECTORS_LOC OUTPUT_DIR
# Try 'python -m spacy init vectors --help' for help.
def load_word_vectors(model_name, word_vectors):
    subprocess.run([sys.executable,
        '-m',
        'spacy',
        'init',
        'vectors',
        'en',
        word_vectors,
        model_name])

# load_word_vectors(model_name, word_vectors)

##########################################################################
# Use the custom toxicity model to train our XGRegressor like in point 2 #
# Compare results using spacy 'en_core_web_lg'                           #
##########################################################################

# Load Ruddit scored comments and clean it
# df = pd.read_csv('./data/ruddit_comments_score.csv')
# clean_body = clean_data(df['body'])
# df_clean = df.iloc[clean_body.index][['score']].copy()
# df_clean['body'] = clean_body.copy()
# df_clean = df_clean.drop([2612,2711], axis=0)
# df_clean.to_csv('./data/ruddit_comments_clean_scored.csv', index=False)
# df_clean.to_excel('./data/ruddit_comments_clean_scored.xls', index=False)

# Load our clean training dataset
df_clean = pd.read_csv('./data/ruddit_comments_clean_scored.csv')

# Get or load a vectorize version of Ruddit DataFrame
# using our custom Word2Vec pipeline.
nlp = spacy.load('./data/toxicity_custom_model')
path = './data/ruddit_custom_w2v.npy'
if os.path.isfile(path):
    df_vectors = np.load(path)
    print('Ruddit comments custom vectorized loaded! Shape: ', df_vectors.shape)
else:
    with nlp.disable_pipes():
        df_vectors = np.array([nlp(text).vector for text in df_clean.body])
        np.save(path, df_vectors)
        print('Ruddit comments custom vectorized saved! Shape: ', df_vectors.shape)

# Get or load a vectorize version of Ruddit DataFrame
nlp = spacy.load('en_core_web_lg')
path = './data/ruddit_spacy_w2v.npy'
if os.path.isfile(path):
    df_vectors_spacy = np.load(path)
    print('Ruddit comments spacy vectorized loaded! Shape: ', df_vectors_spacy.shape)
else:
    with nlp.disable_pipes():
        df_vectors_spacy = np.array([nlp(text).vector for text in df_clean.body])
        np.save(path, df_vectors_spacy)
        print('Ruddit comments spacy vectorized saved! Shape: ', df_vectors_spacy.shape)

# Load validation data
validation_data = pd.read_csv('./data/validation_data.csv')

# Split our dataset based on Ruddits dataset vectorized
X_train, X_test, y_train, y_test = train_test_split(df_vectors, df_clean.score,
                                                    test_size=0.1,
                                                    random_state=1)

# Split our dataset based on Ruddits dataset vectorized
X_train2, X_test2, y_train2, y_test2 = train_test_split(df_vectors_spacy, df_clean.score,
                                                    test_size=0.1,
                                                    random_state=1)

# Using XGBoost training on custom word vector
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
my_model.fit(X_train, y_train, early_stopping_rounds=5,
             eval_set=[(X_test, y_test)], verbose=False)

predictions = my_model.predict(X_test)
mae = mean_absolute_error(predictions, y_test)
print('Mean Absolute Error: ', mae)

# Using XGBoost training on spacy word vector
my_model2 = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
my_model2.fit(X_train2, y_train2, early_stopping_rounds=5,
             eval_set=[(X_test2, y_test2)], verbose=False)

predictions2 = my_model2.predict(X_test2)
mae2 = mean_absolute_error(predictions2, y_test2)
print('Mean Absolute Error: ', mae2)


def clean_valid_data(comment, nlp_model):
    stop = list(nlp_model.Defaults.stop_words)

    comment = comment.lower() # Pass to lower

    pattern = r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"
    comment = re.sub(pattern,"",comment) # Remove unicode

    comment = " ".join([word for word in comment.split() if word not in (stop)]) # Remove stop words

    comment = " ".join([token.lemma_ for token in nlp_model(comment)])

    return comment

def model_score_pairs(comments, model, vector_model, nlp_model):

    less_toxic = comments[0]
    more_toxic = comments[1]

    less_toxic = clean_valid_data(less_toxic, nlp_model)
    more_toxic = clean_valid_data(more_toxic, nlp_model)
    less_vector = vector_model(less_toxic).vector.reshape(1, 300)
    more_vector = vector_model(more_toxic).vector.reshape(1, 300)
    
    less_score = model.predict(less_vector)
    more_score = model.predict(more_vector)
    return pd.Series([less_score, more_score])


def model_predict(validation_data, model, nlp_model, vector_model):

    test_df = validation_data[['less_toxic', 'more_toxic']]
    result = test_df.apply(model_score_pairs, model=model, nlp_model=nlp_model,
        vector_model=vector_model, raw=True, axis=1)
    
    result['value'] = result['less_toxic'] < result['more_toxic']
    return result.value.mean()


# Test our models
val_data_test = validation_data.iloc[:]

nlp_spacy = spacy.load('en_core_web_lg')
mean_score2 = model_predict(val_data_test, model=my_model2, nlp_model=nlp_spacy,
    vector_model=nlp_spacy)
print('Spacy Model', mean_score2)

model_name = './data/toxicity_custom_model'
nlp_custom = spacy.load(model_name)
mean_score = model_predict(val_data_test, model=my_model, nlp_model=nlp_spacy,
    vector_model=nlp_custom)
print('Custom Model', mean_score)
