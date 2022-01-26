import spacy
import pandas as pd
import re
from time import time


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