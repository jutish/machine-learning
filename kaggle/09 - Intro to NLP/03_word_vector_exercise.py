import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier

# Load the large model to get the vectors
# It is necesary to install the model first
# Here are all the available models: https://spacy.io/models/en#en_core_web_lg
nlp = spacy.load('en_core_web_lg')
review_data = pd.read_csv('yelp_ratings.csv')
print(review_data.dtypes)

# Here's an example of loading some document vectors.
# Calculating 44,500 document vectors takes about 20 minutes, so we'll get only
# the first 100. 
reviews = review_data[:100]

# We just want the vectors so we can turn off other models in the pipeline
# The result is a matrix of 100 rows and 300 columns.
with nlp.disable_pipes():
    vectors = np.array([nlp(text).vector for text in reviews.text])
# print(vectors.shape)

# To save time, we'll load pre-saved document vectors for the
# hands-on coding exercises.
vectors = np.load('review_vectors.npy')
# print(vectors.shape)

# 1) Training a Model on Document Vectors
# Next you'll train a LinearSVC model
# using the document vectors. It runs pretty quick and works well in high
# dimensional settings like you have here.

# After running the LinearSVC model, you might try experimenting with other
# types of models to see whether it improves your results.
X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment,
    test_size=0.1, random_state=1)

model = LinearSVC()
model.fit(X_train, y_train)
print(f'LinearSVC score {model.score(X_test, y_test)*100:.3f}%')

#Testing with other model.
model_two = PassiveAggressiveClassifier()
model_two.fit(X_train, y_train)
print(f'PassiveAggressiveClassifier {model_two.score(X_test, y_test)*100:.3f}%')

# Document Similarity
# 2) Centering the Vectors
# Sometimes people center document vectors when calculating similarities. That is,
# they calculate the mean vector from all documents, and they subtract this from
# each individual document's vector. Why do you think this could help with
# similarity metrics?
# Solution: Sometimes your documents will already be fairly similar. For example,
# this data set is all reviews of businesses. There will be stong similarities
# between the documents compared to news articles, technical manuals, and
# recipes. You end up with all the similarities between 0.8 and 1 and no
# anti-similar documents (similarity < 0). When the vectors are centered, you are
# comparing documents within your dataset as opposed to all possible documents.

# 3) Find the most similar review
# Given an example review below, find the most similar document within the Yelp
# dataset using the cosine similarity.

def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

review = """I absolutely love this place. The 360 degree glass windows with the 
Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
transports you to what feels like a different zen zone within the city. I know 
the price is slightly more compared to the normal American size, however the food 
is very wholesome, the tea selection is incredible and I know service can be hit 
or miss often but it was on point during our most recent visit. Definitely recommend!
I would especially recommend the butternut squash gyoza."""

review_vec = nlp(review).vector

# Center the document vectors
# Calculate the mean for the document vectors, should have shape (300,)
vec_mean = vectors.mean(axis=0)

# Subtract the mean from the vectors
centered = vectors - vec_mean

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the review vector
sims = [cosine_similarity((review_vec - vec_mean), center) for center in centered]

# # Get the index for the most similar document
most_similar = np.argmax(sims)
print(review_data.iloc[most_similar].text)