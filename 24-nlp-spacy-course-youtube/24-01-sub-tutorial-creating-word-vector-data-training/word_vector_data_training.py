# How to create a word vector using gensim based on:
# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial/notebook
# We learn who to prepare Text to become a good Data Training for Word Vectors
# using gensim
import numpy as np
import re  # For preprocessing
import pandas as pd  # For data handling
import multiprocessing
import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# sns.set_style("darkgrid")
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", 
#     datefmt= '%H:%M:%S', level=logging.INFO)

# # # Load dataset
# df = pd.read_csv('./sources/simpsons_dataset.csv')

# # # Remove nulls
# print(df.isnull().sum())
# df = df.dropna().reset_index(drop=True)
# print(df.isnull().sum())

# # Cleaning
# # We are lemmatizing and removing the stopwords and non-alphabetic characters 
# # for each line of dialogue.
# nlp = spacy.load('en_core_web_sm', disable=['ner','parser']) # Disabling Named Entity Recognition for speed

# def cleaning(doc):
#     # Lemmatizes and removes stopwords
#     # doc needs to be a spacy Doc object
#     txt = [token.lemma_ for token in doc if not token.is_stop]
#     if len(txt) > 2:
#         return ' '.join(txt)

# Removes non-alphabetic characters:
# We create a function called brief_cleaning (generator class) which call
# allow you to declare a function that behaves like an iterator, 
# i.e. it can be used in a for loop just once. This function iterate over the dialogs, 
# converts it to lowercase and replace non-alphabetic characters with spaces ' '
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])

# # # Taking advantage of spaCy .pipe() attribute to speed-up the cleaning process:
# t = time()
# txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
# print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

# # Put the results in a DataFrame to remove missing values and duplicates:
# df_clean = pd.DataFrame({'clean': txt})
# df_clean = df_clean.dropna().drop_duplicates()
# print(df_clean.shape)

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
# bigram = Phraser(phrases)

# # Transform the corpus based on the bigrams detected:
# sentences = bigram[sent]
# print(type(sentences))

file_name = 'simpson_data.mm'
# pickle.dump(sentences, open(file_name, 'wb'))
sentences = pickle.load(open(file_name, 'rb'))



# # Most Frequent Words: Mainly a sanity check of the effectiveness of the
# # lemmatization, removal of stopwords, and addition of bigrams.
# word_freq = defaultdict(int)
# for sent in sentences:
#     for i in sent:
#         word_freq[i] += 1

# # Show top 10 of most frecuent words.
# print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

# # Training the model 
# # Gensim Word2Vec Implementation: 
# # We use Gensim implementation
# # of word2vec: https://radimrehurek.com/gensim/models/word2vec.html

# # Why I seperate the training of the model in 3 steps: I prefer to separate the
# # training in 3 distinctive steps for clarity and monitoring.

# # Word2Vec(): 
# # In this first step, I set up the parameters of the model
# # one-by-one. I do not supply the parameter sentences, and therefore leave the
# # model uninitialized, purposefully.

# # .build_vocab(): 
# # Here it builds the vocabulary from a sequence of sentences and
# # thus initialized the model. With the loggings, I can follow the progress and
# # even more important, the effect of min_count and sample on the word corpus. I
# # noticed that these two parameters, and in particular sample, have a great
# # influence over the performance of a model. Displaying both allows for a more
# # accurate and an easier management of their influence.

# # .train(): 
# # Finally, trains the model. The loggings here are mainly useful for
# #  monitoring, making sure that no threads are executed instantaneously.

# cores = multiprocessing.cpu_count() # Count the number of cores in a computer - 4 in my case

# # The parameters:

# # min_count = int - Ignores all words with total absolute frequency lower than
# # this - (2, 100)

# # window = int - The maximum distance between the current and predicted word
# # within a sentence. E.g. window words on the left and window words on the left
# # of our target - (2, 10)

# # size = int - Dimensionality of the feature vectors. - (50, 300)

# # sample = float - The threshold for configuring which higher-frequency words are
# # randomly downsampled. Highly influencial. - (0, 1e-5)

# # alpha = float - The initial learning rate - (0.01, 0.05)

# # min_alpha = float - Learning rate will linearly drop to min_alpha as training
# # progresses. To set it: alpha - (min_alpha * epochs) ~ 0.00

# # negative = int - If > 0, negative sampling will be used, the int for negative
# # specifies how many "noise words" should be drown. If set to 0, no negative
# # sampling is used. - (5, 20)

# # workers = int - Use these many worker threads to train the model
# # (=faster training with multicore machines)

# # Step 1
# w2v_model = Word2Vec(min_count=20,
#                      window=2,
#                      vector_size=300,
#                      sample=6e-5, 
#                      alpha=0.03, 
#                      min_alpha=0.0007, 
#                      negative=20,
#                      workers=cores-1)

# # Step 2
# # Building the Vocabulary Table: Word2Vec requires us to build the vocabulary
# # table (simply digesting all the words and filtering out the unique words, and
# # doing some basic counts on them):
# t = time()
# w2v_model.build_vocab(sentences, progress_per=10000)
# print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# # Step 3
# # Training of the model:
# # Parameters of the training:
# # total_examples = int - Count of sentences;
# # epochs = int - Number of iterations (epochs) over the corpus - [10, 20, 30]
# t = time()
# w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
# print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# # As we do not plan to train the model any further, we are calling init_sims()
# # which will make the model much more memory-efficient:
# w2v_model.init_sims()
# w2v_model.save('./sources/word2vec_model')

# Exploring the model Most similar to: 
# Here, we will ask our model to find the
# word most similar to some of the most iconic characters of the Simpsons!

model = Word2Vec.load('./sources/word2vec_model')
# similar_words = model.wv.most_similar(positive=["homer"])
# print(similar_words)

# Let's see what the bigram "homer_simpson" gives us by comparison:
# similar_words = model.wv.most_similar(positive=['homer_simpson'])
# print(similar_words)

# What about Marge now?
# similar_words = model.wv.most_similar(positive=["marge"])
# print(similar_words)

# Let's check Bart now:
# print(model.wv.most_similar(positive=['bart']))

# # Willie the groundskeeper for the last one:
# print(model.wv.most_similar(positive=['willie']))

# Similarities:
# Here, we will see how similar are two words to each other :
# print(model.wv.similarity("moe", 'tavern'))
# print(model.wv.similarity('maggie', 'baby'))

# Odd-One-Out:
# Here, we ask our model to give us the word that does not belong to the list!
# Between Jimbo, Milhouse, and Kearney, who is the one who is not a bully?
# print(model.wv.doesnt_match(['jimbo', 'milhouse', 'kearney']))

# What if we compared the friendship between Nelson, Bart, and Milhouse?
# print(model.wv.doesnt_match(['bart','milhouse','nelson']))

# Analogy difference:
# Which word is to woman as homer is to marge?
# rs = model.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3)
# print(rs)
# # Which word is to woman as bart is to man?
# rs = model.wv.most_similar(positive=['bart','woman'], negative=['man'])
# print(rs)

# t-SNE visualizations: 
# t-SNE is a non-linear dimensionality reduction algorithm
# that attempts to represent high-dimensional data and the underlying
# relationships between vectors in a lower-dimensional space. Here is a good
# tutorial on it:
# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

# Our goal in this section is to plot our 300 dimensions vectors into 2
# dimensional graphs, and see if we can spot interesting patterns. For that we
# are going to use t-SNE implementation from scikit-learn.

# To make the visualizations more relevant, we will look at the relationships
# between a query word (in **red**), its most similar words in the model
# (in **blue**), and other words from the vocabulary (in **green**).

def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction 
    algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.show()

# 10 Most similar words vs. 8 Random words: 
# Let's compare where the vector
# representation of Homer, his 10 most similar words from the model, as well as 8
# random ones, lies in a 2D graph:

# # Interestingly, the 10 most similar words to Homer ends up around him, so does
# # Apu and (sideshow) Bob, two recurrent characters.
# tsnescatterplot(model, 'homer', 
#     ['dog', 'bird', 'ah', 'maude', 'bob', 'mel', 'apu', 'duff'])

# # 10 Most similar words vs. 10 Most dissimilar 
# # This time, let's compare where the
# # vector representation of Maggie and her 10 most similar words from the model
# # lies compare to the vector representation of the 10 most dissimilar words to
# # Maggie:
# # Neat! Maggie and her most similar words form a distinctive cluster from the most
# # dissimilar words, it is a really encouraging plot! 
# tsnescatterplot(model, 'maggie',[i[0] for i in model.wv.most_similar(negative=["maggie"])])

