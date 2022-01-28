# key concepts
# Topics
# Clusterings
# K-Means
# Bi-Grams
# Three-Grams
# TF-IDF (Term Frecuency - Inverse Document Frecuency)

# Supose we have 10.000 documents each one with 100 Wo#rds and we want
# to know the TF-IDF for the "Violence" word in the first document.

# So TF = Times tha Violence appears in the first document over the 
# the number of word in that document. Eg. 35 Times / 100 Words = 0.35

# IDF = log(number of total documents / 
# number of documents where "Violence" appears)
# Eg. log(10000 / 350) = 1.45

# TF-IDF = TF * IDF = 0.35 * 1.45 = 0.5

import pandas as pd
import string
import json
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords
from time import time

def load_data(file):
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file,'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Load data
names = load_data('./data/trc_dn.json')['names'][1]
descriptions = load_data('./data/trc_dn.json')['descriptions']


# 
def remove_stops(text, stops):
    text = re.sub(r"AC\/\d{1,4}\/\d{1,4}", "", text)  # Remove (AC/2000/142)
    text = ' '.join([word for word in text.split() if word not in stops]) # Remove months and stopwords
    text = text.translate(text.maketrans(" ", " ", string.punctuation)) # Remove punctuation
    text = ''.join([word for word in text if not word.isdigit()]) # Remove numbers
    while '  ' in text: # Remove double spaces
        text = text.replace('  ',' ')
    return text

# Clean Data
def clean_docs(docs):
    t = time()
    stops = stopwords.words('english')
    months = load_data('./data/months.json')
    stops = stops + months
    final = []
    for doc in docs:
        final.append(remove_stops(doc, stops))
    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    return final

# Clean our descriptions
cleaned_docs = clean_docs(descriptions)

# Instance our TF-IDF vectorizer
vectorizer = TfidfVectorizer(
                                lowercase=True,
                                max_features=100,
                                max_df=0.8,
                                min_df=5,
                                ngram_range=(1, 3),
                                stop_words = 'english'
                            )
vectors = vectorizer.fit_transform(cleaned_docs)

# Get all the words in our vocabulary
feature_names = vectorizer.get_feature_names_out()

# Pass from the sparce matrix representation to the dense representation
# Dense rep. is just the matrix with all the rows and columns
# The sparce rep. is a way of use less memory. and rep. the matrix with
# tuples.
dense = vectors.todense()
denselist = dense.tolist()  #Each row is a vector TF-IDF of ours descriptions

# For each vector/text make a list with only the words which have a TF-IDF
# greater than 0
all_keywords = []
for description_vec in denselist:
    x = 0
    keywords = []
    for word_tfid_val in description_vec:
        if word_tfid_val > 0:
            keywords.append(feature_names[x])
        x += 1
        all_keywords.append(keywords)

print(descriptions[0:1])
print(all_keywords[0:1])

# Cluster using K-Means
true_k = 20 # Number of clusters to get
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100,
    n_init=1)
model.fit(vectors)
centroids = model.cluster_centers_
order_centroids = model.cluster_centers_.argsort()[:, ::-1]  # returns the index ordered in DESC mode.
terms = vectorizer.get_feature_names_out()

# Write our cluster results
with open('./data/trc_results.txt', 'w', encoding='utf-8') as f:
    for i in range(true_k):
        f.write(f'Cluster {i}')
        f.write('\n')
        for ind in order_centroids[i, :10]:
            # f.write(f'{terms[ind]} {centroids[i, ind]}')
            f.write(f'{terms[ind]}')
            f.write('\n')
        f.write('\n')
        f.write('\n')

# Plotting our results
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get the cluster for each row return an ndarray
kmeans_indices = model.fit_predict(vectors)

# Reduce dimension of tf-idf vectors and plot it
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())
colors = ['r','b','c','y','m']  # We have one color per cluster
x_axis = [reduced_row[0] for reduced_row in scatter_plot_points]
y_axis = [reduced_row[1] for reduced_row in scatter_plot_points]
fig, ax = plt.subplots(50,50)
ax.scatter(x_axis, y_axis, c=[colors[i] for i in kmeans_indices])
for i, name in enumerate(names):
    ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))
plt.savefig('trc.png')