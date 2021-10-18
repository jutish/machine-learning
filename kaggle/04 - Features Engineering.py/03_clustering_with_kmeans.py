# Adding a feature of cluster labels can help machine learning models untangle 
# complicated relationships of space or proximity.
# The motivating idea for adding cluster labels is that the clusters will break 
# up complicated relationships across features into simpler chunks. 
# Our model can then just learn the simpler chunks one-by-one instead having 
# to learn the complicated whole all at once. It's a "divide and conquer" strategy

# k-Means Clustering There are a great many clustering algorithms. They differ
# primarily in how they measure "similarity" or "proximity" and in what kinds of
# features they work with. The algorithm we'll use, k-means, is intuitive and
# easy to apply in a feature engineering context. Depending on your application
# another algorithm might be more appropriate.

# K-Means

# It's a simple two-step process. The algorithm starts by randomly initializing
# some predefined number (n_clusters) of centroids. It then iterates over these
# two operations:

# assign points to the nearest cluster centroid move each centroid to minimize the
# distance to its points It iterates over these two steps until the centroids
# aren't moving anymore, or until some maximum number of iterations has passed
# (max_iter).

# It often happens that the initial random position of the centroids ends in a
# poor clustering. For this reason the algorithm repeats a number of times
# (n_init) and returns the clustering that has the least total distance between
# each point and its centroid, the optimal clustering.

# Example - California Housing
# As spatial features, California Housing's 'Latitude' and 'Longitude' make
# natural candidates for k-means clustering. In this example we'll cluster
# these with 'MedInc' (median income) to create economic segments in different
# regions of California.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# Load data
df = pd.read_csv('./recursos/housing.csv')
X = df.copy()
y = X.pop('median_house_value')

X_features = df[['median_income','latitude','longitude']].copy()

# Define score function and model
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(model, X, y=y, cv=5, 
        scoring='neg_median_absolute_error', verbose=True, error_score=0)
    score = -1 * score.mean()
    return score

# Create cluster future 
# Since k-means clustering is sensitive to scale, it can
# be a good idea rescale or normalize data with extreme values. 
# Formula to scale: (X - X.mean(axis=0)) / X.std(axis=0)
# Formula to escale between 0 and 1: (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) This gets betters results!
# Our features
# are already roughly on the same scale, so we'll leave them as-is.
k_means = KMeans(n_clusters=6, n_init=10)
X['Cluster'] = k_means.fit_predict(X_features)
X['Cluster'] = X['Cluster'].astype('category')
# print(X.sort_values(by='median_income', ascending=False))

# Now let's look at a couple plots to see how effective this was. First, a
# scatter plot that shows the geographic distribution of the clusters. It seems
# like the algorithm has created separate segments for higher-income areas on
# the coasts.
# sns.relplot(x='longitude', y='latitude', hue='Cluster', data=X)
# plt.show()

# The target in this dataset is MedHouseVal (median house value). These
# box-plots show the distribution of the target within each cluster. If the
# clustering is informative, these distributions should, for the most part,
# separate across MedHouseVal, which is indeed what we see.

# sns.catplot(x='median_house_value', y='Cluster', data=pd.concat([X,df['median_house_value']],axis=1), kind='boxen')
# plt.show()

# # Score without cluster
X_without_score = X.drop(['Cluster'], axis=1).copy()
score_cluster = score_dataset(X_without_score, y)
print('Score Without Cluster:', score_cluster)
# # # Score model with Cluster column
score_cluster = score_dataset(X, y)
print('Score Cluster:', score_cluster)

# 3) Cluster-Distance Features The k-means algorithm offers an alternative way
# of creating features. Instead of labelling each feature with the nearest
# cluster centroid, it can measure the distance from a point to all the
# centroids and return those distances as features. Now add the
# cluster-distance features to your dataset. You can get these distance
# features by using the fit_transform method of kmeans instead of fit_predict.
k_means = KMeans(n_clusters=6, n_init=10)
transform = k_means.fit_transform(X_features)
transform = pd.DataFrame(transform, columns=[f'Centroid_{i}' for i in range(transform.shape[1])])
X = X.join(transform).drop('Cluster', axis=1)
score_cluster = score_dataset(X, y)
print('Score Cluster transform:', score_cluster)
