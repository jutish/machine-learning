# Import libraries
import nltk
nltk.download('stopwords')
import numpy as np
import json
import glob # The glob module finds all the pathnames matching a specified pattern
import gensim
import gensim.corpora as corpora
import spacy
import pyLDAvis
import pyLDAvis.gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

# Silence Warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:03:53.520589Z","iopub.execute_input":"2022-02-22T00:03:53.521278Z","iopub.status.idle":"2022-02-22T00:03:53.531133Z","shell.execute_reply.started":"2022-02-22T00:03:53.521228Z","shell.execute_reply":"2022-02-22T00:03:53.529990Z"}}
# Preparing data
def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        jsond.dump(data, f, indent=4)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:03:53.532460Z","iopub.execute_input":"2022-02-22T00:03:53.532925Z","iopub.status.idle":"2022-02-22T00:03:53.558753Z","shell.execute_reply.started":"2022-02-22T00:03:53.532871Z","shell.execute_reply":"2022-02-22T00:03:53.557729Z"}}
# get stopwords
stopwords = stopwords.words('english')
print(stopwords)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:03:53.561261Z","iopub.execute_input":"2022-02-22T00:03:53.561790Z","iopub.status.idle":"2022-02-22T00:03:53.860480Z","shell.execute_reply.started":"2022-02-22T00:03:53.561752Z","shell.execute_reply":"2022-02-22T00:03:53.859586Z"}}
# load data
data = load_data('../input/ushmm-dn/ushmm_dn.json')['texts']
print(data[0][0:90])

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:07:19.944543Z","iopub.execute_input":"2022-02-22T00:07:19.945075Z","iopub.status.idle":"2022-02-22T00:12:13.152887Z","shell.execute_reply.started":"2022-02-22T00:07:19.945038Z","shell.execute_reply":"2022-02-22T00:12:13.150946Z"}}
def lemmatization(texts, allowed_postags=['NOUN','ADJ','VERB','ADV']):
    nlp = spacy.load('en_core_web_sm')
    output = [' '.join([word.lemma_ for word in text if word.pos_ in allowed_postags]) 
              for text in list(nlp.pipe(texts, batch_size=10, disable=['parser','ner']))]
    return output

lemmatized_texts = lemmatization(data[:])
print(lemmatized_texts[0][0:90])

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:12:13.154946Z","iopub.execute_input":"2022-02-22T00:12:13.155188Z","iopub.status.idle":"2022-02-22T00:12:17.383432Z","shell.execute_reply.started":"2022-02-22T00:12:13.155159Z","shell.execute_reply":"2022-02-22T00:12:17.382018Z"}}
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)  # Remove accents
        final.append(new)
    return final

data_words = gen_words(lemmatized_texts)
print(data_words[0][0:90])

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:14:25.068730Z","iopub.execute_input":"2022-02-22T00:14:25.069060Z","iopub.status.idle":"2022-02-22T00:14:34.316843Z","shell.execute_reply.started":"2022-02-22T00:14:25.069028Z","shell.execute_reply":"2022-02-22T00:14:34.315896Z"}}
# Bigrams and Trigrams

# Instance and train a Phrases object to Automatically detect common phrases – 
# aka multi-word expressions, word n-gram collocations – from a stream of sentences.
bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100, 
                                       connector_words=ENGLISH_CONNECTOR_WORDS)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], min_count=5, 
                                       threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)  # It's a wrapper for Phrases objects
trigram = gensim.models.phrases.Phraser(trigram_phrases) # it is a faster way to use it

data_bigrams = bigram[data_words] # get bigrams
data_bigrams_trigrams = trigram[data_bigrams] #get trigrams from bigrams

print(data_bigrams_trigrams[0][0:90])

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:14:34.318437Z","iopub.execute_input":"2022-02-22T00:14:34.318689Z","iopub.status.idle":"2022-02-22T00:14:47.001067Z","shell.execute_reply.started":"2022-02-22T00:14:34.318659Z","shell.execute_reply":"2022-02-22T00:14:46.999967Z"}}
#TF-IDF REMOVAL
# How to filter out words with low tf-idf in a corpus with gensim?

from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words  = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = [] #reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:07:09.969582Z","iopub.status.idle":"2022-02-22T00:07:09.970984Z","shell.execute_reply.started":"2022-02-22T00:07:09.970598Z","shell.execute_reply":"2022-02-22T00:07:09.970635Z"}}
# # Create a list where each text is a list of tuples like [[(0,5),(2,10),...,(350,10)],[...]]
# # Each tuple is (word_id, frecuency_on_all_the_texts)
# # See doc here: https://radimrehurek.com/gensim/corpora/dictionary.html

# id2word = corpora.Dictionary(data_words)  # Create a dictionary where each word have an id
# corpus = []
# for text in data_words:
#     new = id2word.doc2bow(text)
#     corpus.append(new)

# print(corpus[0][0:90])

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:14:47.002694Z","iopub.execute_input":"2022-02-22T00:14:47.002926Z","iopub.status.idle":"2022-02-22T00:14:51.770946Z","shell.execute_reply.started":"2022-02-22T00:14:47.002898Z","shell.execute_reply":"2022-02-22T00:14:51.770019Z"}}
# Create our LDA model
# Use all the texts, except the last one, to train our model.
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                            id2word=id2word, 
                                            num_topics=15,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto')

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:41:15.525317Z","iopub.execute_input":"2022-02-22T00:41:15.525661Z","iopub.status.idle":"2022-02-22T00:41:15.535013Z","shell.execute_reply.started":"2022-02-22T00:41:15.525625Z","shell.execute_reply":"2022-02-22T00:41:15.534080Z"}}
# Apply the model to an unsee text.
test_doc = corpus[-1]
vector = lda_model[test_doc] # Return a vector with the prob. of test_doc belongs to each category

def order(vector):
    vector.sort(reverse=True,key = lambda x: x[1])
    return vector

new_vec = order(vector)
print(new_vec)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:51:24.228506Z","iopub.execute_input":"2022-02-22T00:51:24.228839Z","iopub.status.idle":"2022-02-22T00:51:24.285821Z","shell.execute_reply.started":"2022-02-22T00:51:24.228791Z","shell.execute_reply":"2022-02-22T00:51:24.285050Z"}}
# Save the model
lda_model.save('our_model.model')

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:52:39.925269Z","iopub.execute_input":"2022-02-22T00:52:39.925731Z","iopub.status.idle":"2022-02-22T00:52:39.948104Z","shell.execute_reply.started":"2022-02-22T00:52:39.925690Z","shell.execute_reply":"2022-02-22T00:52:39.947148Z"}}
# Load the model
our_model = gensim.models.ldamodel.LdaModel.load('our_model.model')

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:53:32.669522Z","iopub.execute_input":"2022-02-22T00:53:32.669856Z","iopub.status.idle":"2022-02-22T00:53:32.680029Z","shell.execute_reply.started":"2022-02-22T00:53:32.669813Z","shell.execute_reply":"2022-02-22T00:53:32.679020Z"}}
# Test our loaded model
test_doc = corpus[-1]
vector = our_model[test_doc] # Return a vector with the prob. of test_doc belongs to each category

def order(vector):
    vector.sort(reverse=True,key = lambda x: x[1])
    return vector

new_vec = order(vector)
print(new_vec)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T00:07:09.975231Z","iopub.status.idle":"2022-02-22T00:07:09.975571Z","shell.execute_reply.started":"2022-02-22T00:07:09.975396Z","shell.execute_reply":"2022-02-22T00:07:09.975413Z"}}
# Visualizing the data
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=30)
vis