# https://aqsazafar81.medium.com/10-data-science-projects-for-beginners-to-sharpen-skills-in-2021-5cffa4f369dd
# https://www.datacamp.com/community/tutorials/scikit-learn-fake-news/?tap_a=5644-dce66f&tap_s=950491-315da1&utm_medium=affiliate&utm_source=aqsazafar
# https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier

# Read data
path = 'fake_or_real_news.csv'
df = pd.read_csv(path, index_col= 'Unnamed: 0')
print('Shape:', df.shape)
print('Columns:', df.columns)
print('First Rows\n', df.head())

# Extracting the training data 
# What is a TfidfVectorizer? 
# TF (Term Frequency):
# The number of times a word appears in a document is its Term Frequency. A
# higher value means a term appears more often than others, and so, the
# document is a good match when the term is part of the search terms.
# IDF (Inverse Document Frequency): 
# Words that occur many times a document, but
# also occur many times in many others, may be irrelevant. IDF is a measure of
# how significant a term is in the entire corpus.
# The TfidfVectorizer converts a collection of raw documents into a matrix of
# TF-IDF features.
y = df.pop('label')
X = df['text'].copy()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
    test_size=0.33, random_state=0)
print('Train Shape: ', X_train.shape, 'Train label shape:', y_train.shape)
print('Test Shape: ', X_test.shape, 'Test label shape:', y_test.shape)

# Building Vectorizer Classifiers 
# Term Frecuency - Invert Document Frecuency AND Count Vectorizer
# Now that you have your training and testing
# data, you can build your classifiers. To get a good idea if the words and
# tokens in the articles had a significant impact on whether the news was fake
# or real, you begin by using CountVectorizer and TfidfVectorizer.
# You’ll see the example has a max threshhold set at .7 for the TF-IDF
# vectorizer tfidf_vectorizer using the max_df argument. This removes words
# which appear in more than 70% of the articles. Also, the built-in stop_words
# parameter will remove English stop words from the data before making vectors.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Now that you have vectors, you can then take a look at the vector features,
# stored in count_vectorizer and tfidf_vectorizer.
print(tfidf_vectorizer.get_feature_names())
print(count_vectorizer.get_feature_names())

# Comparing Models
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
print(count_df.head())
print(tfidf_df.head())

# Use a NLP model MultinomialNB on tfidf_vectorizer
clf = MultinomialNB()
clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=['REAL','FAKE'])
print('MultinomialNB Accuracy Score: %0.3f' % score)
print('MultinomialNB Confussion matrix: \n', cm)

# Use a NLP model MultinomialNB on count_vectorizer
clf = MultinomialNB()
clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=['REAL','FAKE'])
print('MultinomialNB Accuracy Score on CountVectorizer: %0.3f' % score)
print('MultinomialNB Confussion matrix on CountVectorizer: \n', cm)

# What is a PassiveAggressiveClassifier?
# Passive Aggressive algorithms are online learning algorithms. Such an
# algorithm remains passive for a correct classification outcome, and turns
# aggressive in the event of a miscalculation, updating and adjusting. Unlike
# most other algorithms, it does not converge. Its purpose is to make updates
# that correct the loss, causing very little change in the norm of the weight
# vector.

# Testing Linear Models - Passive Aggresive Classifier - on tfidf_vectorizer
linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print("Passive Aggresive Classifier - on tfidf_vectorizer: %0.3f" % score)
print('Passive Aggresive Classifier Confussion matrix on tfidf_vectorizer: \n', cm)

# Testing Linear Models - Passive Aggresive Classifier - on count_vectorizer
linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(count_train, y_train)
pred = linear_clf.predict(count_test)
score = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print("Passive Aggresive Classifier - on count_vectorizer: %0.3f" % score)
print('Passive Aggresive Classifier Confussion matrix on count_vectorizer: \n', cm)