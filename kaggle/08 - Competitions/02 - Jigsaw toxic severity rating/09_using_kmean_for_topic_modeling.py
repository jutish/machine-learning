import pandas as pd
import string
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords
from time import time


# Load ruddit scored comments
ruddit_comments_score = pd.read_csv('./data/ruddit_comments_clean_scored.csv', usecols=['score'])

# Load spacy vectorized version of ruddits scored comments
ruddit_vectorized_body = np.load('./data/ruddit_spacy_w2v.npy')

# Vamos a hallar el valor de K haciendo una gráfica e intentando hallar el
# “punto de codo” que comentábamos antes --> https://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/
# Este es nuestro resultado:
# Nc = range(1, 20)
# # Creo un array de objetos KMeans con distintos tamaños de Clusters
# kmeans = [KMeans(n_clusters=i) for i in Nc]
# score = [kmeans[i].fit(ruddit_vectorized_body).score(ruddit_vectorized_body) 
#          for i in range(len(kmeans))]
# plt.plot(Nc, score)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show()

# Cluster using K-Means
true_k = 5 # Number of clusters to get
model = KMeans(n_clusters=true_k)
cluster_per_row = model.fit_predict(ruddit_vectorized_body)

# Reduce dimension of vectors and plot it
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(ruddit_vectorized_body)
colors = ['r','b','c','y','m']  # We have one color per cluster
x_axis = [reduced_row[0] for reduced_row in scatter_plot_points]
y_axis = [reduced_row[1] for reduced_row in scatter_plot_points]
fig, ax = plt.subplots(figsize=(50,50))
ax.scatter(x_axis, y_axis, c=[colors[i] for i in cluster_per_row])
for i, score in enumerate(ruddit_comments_score.to_numpy()):
    ax.annotate(f'{score}', (x_axis[i], y_axis[i]))
plt.savefig('./data/spacy_w2v.png')