# Load libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
import spacy
from time import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from sklearn import svm

# Get stopwords from nltk
stop_words = stopwords.words('english')

# Load training dataset
ruddit_scored = pd.read_csv('./data/ruddit_comments_score.csv')
print('Ruddit Scored Loaded: ', ruddit_scored.shape)

# # Spacy Word2Vec
# # Convert our ruddit_scored into a Word2Vec representation using spacy
# nlp = spacy.load('en_core_web_lg')
# # Use the "pipe" way, it's better for larger Texts (almost 50% better)
# ruddit_scored_wv = np.array([text.vector for text in 
#     nlp.pipe(ruddit_scored.body, 
#         disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])])

# Sentence Similarity using all-mpnet-base-v2
# https://www.pinecone.io/learn/dense-vector-embeddings-nlp/
# pip install sentence-transformers

# from sentence_transformers import SentenceTransformer
# st = SentenceTransformer('all-mpnet-base-v2')
# X_st = st.encode(ruddit_scored.body, convert_to_numpy=True, show_progress_bar=True)
# np.save('./data/ruddit_scored_body_s2v', X_st)
# print(X_st.shape)
X_st = np.load('./data/ruddit_scored_body_s2v')

# # # Split our data
# y = ruddit_scored.pop('score')
# X = ruddit_scored['body'].copy()

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, 
#     test_size=0.1, shuffle=True)

# X_train_vec, X_test_vec, y_train, y_test = train_test_split(ruddit_scored_wv, y,
#     random_state=0, test_size=0.1, shuffle=True)

# # TF-IDF Always start with these features. They work (almost) everytime!
# tfv = TfidfVectorizer(min_df=3,  max_features=None, 
#             strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#             ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
#             stop_words = 'english')
# tfv.fit(list(X_train) + list(X_test))
# X_train_tf = tfv.transform(X_train)
# X_test_tf = tfv.transform(X_test)

# # Count Vectorizer (Bag-of-words)
# ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
#             ngram_range=(1, 3), stop_words = 'english')
# ctv.fit(list(X_train) + list(X_test))
# X_train_ctv = ctv.transform(X_train)
# X_test_ctv = ctv.transform(X_test)

# # Using XGBoost training on TF-IDF
# xgb_regressor = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
# xgb_regressor.fit(X_train_tf, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test_tf, y_test)], verbose=False)
# predictions = xgb_regressor.predict(X_test_tf)
# mae = mean_absolute_error(predictions, y_test)
# print('XGB Regressor on TF-IDF - MAE: ', mae)

# Using XGBoost training on CountVectorizer
# xgb_regressor_ctv = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
# xgb_regressor_ctv.fit(X_train_ctv, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test_ctv, y_test)], verbose=False)
# predictions = xgb_regressor_ctv.predict(X_test_ctv)
# mae = mean_absolute_error(predictions, y_test)
# print('XGB Regressor on CountVectorizer - MAE: ', mae)

# Using XGBoost training on Word2Vec dataset
# xgb_regressor_w2v = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
# xgb_regressor_w2v.fit(X_train_vec, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test_vec, y_test)], verbose=False)
# predictions = xgb_regressor_w2v.predict(X_test_vec)
# mae = mean_absolute_error(predictions, y_test)
# print('XGB Regressor on Word2Vec - MAE: ', mae)

# # Scale data before use SVM
# scl_tfv = preprocessing.StandardScaler(with_mean=False)
# scl_ctv = preprocessing.StandardScaler(with_mean=False)
# scl_w2v = preprocessing.StandardScaler()

# scl_tfv.fit(X_train_tf)
# scl_ctv.fit(X_train_ctv)
# scl_w2v.fit(X_train_vec)

# X_train_tfv_scl = scl_tfv.transform(X_train_tf)
# X_test_tfv_scl = scl_tfv.transform(X_test_tf)

# X_train_ctv_scl = scl_ctv.transform(X_train_ctv)
# X_test_ctv_scl = scl_ctv.transform(X_test_ctv)

# X_train_w2v_scl = scl_w2v.transform(X_train_vec)
# X_test_w2v_scl = scl_w2v.transform(X_test_vec)

# # Using SVM on TF-IDF
# svm_tfv = svm.SVR()
# svm_tfv.fit(X_train_tf, y_train)
# predictions = svm_tfv.predict(X_test_tf)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-Svr on TF-IDF - MAE: ', mae)

# # Using SVM on CountVectorizer
# svm_tfv = svm.SVR()
# svm_tfv.fit(X_train_ctv, y_train)
# predictions = svm_tfv.predict(X_test_ctv)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-Svr on CountVectorizer - MAE: ', mae)

# # Using SVM on Word2Vec dataset
# svm_tfv = svm.SVR()
# svm_tfv.fit(X_train_vec, y_train)
# predictions = svm_tfv.predict(X_test_vec)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-Svr  on Word2Vec - MAE: ', mae)

# # Using SVM on TF-IDF Scaled
# svm_tfv = svm.SVR()
# svm_tfv.fit(X_train_tfv_scl, y_train)
# predictions = svm_tfv.predict(X_test_tfv_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-Svr Scaled on TF-IDF - MAE: ', mae)

# # Using SVM on CountVectorizer Scaled
# svm_tfv = svm.SVR()
# svm_tfv.fit(X_train_ctv_scl, y_train)
# predictions = svm_tfv.predict(X_test_ctv_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-Svr Scaled on CountVectorizer - MAE: ', mae)

# # Using SVM on Word2Vec dataset Scaled
# svm_tfv = svm.SVR()
# svm_tfv.fit(X_train_w2v_scl, y_train)
# predictions = svm_tfv.predict(X_test_w2v_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-Svr Scaled on Word2Vec - MAE: ', mae)

# # Using SVM on TF-IDF
# svm_tfv = svm.NuSVR()
# svm_tfv.fit(X_train_tf, y_train)
# predictions = svm_tfv.predict(X_test_tf)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-NuSVR on TF-IDF - MAE: ', mae)

# # Using SVM on CountVectorizer
# svm_tfv = svm.NuSVR()
# svm_tfv.fit(X_train_ctv, y_train)
# predictions = svm_tfv.predict(X_test_ctv)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-NuSVR on CountVectorizer - MAE: ', mae)

# # Using SVM on Word2Vec dataset
# svm_tfv = svm.NuSVR()
# svm_tfv.fit(X_train_vec, y_train)
# predictions = svm_tfv.predict(X_test_vec)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-NuSVR on Word2Vec - MAE: ', mae)

# # Using SVM on TF-IDF Scaled
# svm_tfv = svm.NuSVR()
# svm_tfv.fit(X_train_tfv_scl, y_train)
# predictions = svm_tfv.predict(X_test_tfv_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-NuSVR Scaled on TF-IDF - MAE: ', mae)

# # Using SVM on CountVectorizer Scaled
# svm_tfv = svm.NuSVR()
# svm_tfv.fit(X_train_ctv_scl, y_train)
# predictions = svm_tfv.predict(X_test_ctv_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-NuSVR Scaled on CountVectorizer - MAE: ', mae)

# # Using SVM on Word2Vec dataset Scaled
# svm_tfv = svm.NuSVR()
# svm_tfv.fit(X_train_w2v_scl, y_train)
# predictions = svm_tfv.predict(X_test_w2v_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-NuSVR Scaled on Word2Vec - MAE: ', mae)

# # Using SVM LinearSVR on TF-IDF
# svm_tfv = svm.LinearSVR(max_iter=10000)
# svm_tfv.fit(X_train_tf, y_train)
# predictions = svm_tfv.predict(X_test_tf)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-LinearSVR on TF-IDF - MAE: ', mae)

# # Using SVM LinearSVR on CountVectorizer
# svm_tfv = svm.LinearSVR(max_iter=10000)
# svm_tfv.fit(X_train_ctv, y_train)
# predictions = svm_tfv.predict(X_test_ctv)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-LinearSVR on CountVectorizer - MAE: ', mae)

# # Using SVM LinearSVR on Word2Vec dataset
# svm_tfv = svm.LinearSVR(max_iter=10000)
# svm_tfv.fit(X_train_vec, y_train)
# predictions = svm_tfv.predict(X_test_vec)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-LinearSVR on Word2Vec - MAE: ', mae)

# # Using SVM LinearSVR on TF-IDF Scaled
# svm_tfv = svm.LinearSVR(max_iter=50000)
# svm_tfv.fit(X_train_tfv_scl, y_train)
# predictions = svm_tfv.predict(X_test_tfv_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-LinearSVR Scaled on TF-IDF - MAE: ', mae)

# # Using SVM on CountVectorizer
# svm_tfv = svm.LinearSVR(max_iter=50000)
# svm_tfv.fit(X_train_ctv_scl, y_train)
# predictions = svm_tfv.predict(X_test_ctv_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-LinearSVR Scaled on CountVectorizer - MAE: ', mae)

# # Using SVM on Word2Vec dataset
# svm_tfv = svm.LinearSVR(max_iter=10000)
# svm_tfv.fit(X_train_w2v_scl, y_train)
# predictions = svm_tfv.predict(X_test_w2v_scl)
# mae = mean_absolute_error(predictions, y_test)
# print('SVM-LinearSVR Scaled on Word2Vec - MAE: ', mae)





# # Validate our model
# def model_score_pairs(less_toxic, more_toxic, model):
#     less_score = model.predict(less_toxic)
#     more_score = model.predict(more_toxic)
#     return {'less_toxic':less_score[0], 'more_toxic':more_score[0]}


# def model_predict(less_toxic, more_toxic, model):
#     result = pd.DataFrame(columns=['less_toxic', 'more_toxic'])
#     for i in tqdm(range(less_toxic.shape[0])):
#         model_score = model_score_pairs(less_toxic[i], more_toxic[i], model)
#         result = result.append(model_score, ignore_index=True)
#     result['value'] = result['less_toxic'] < result['more_toxic']
#     return result.value.mean()


# # Transform our validation data using our TF-IDF Instance.
# validation_data = pd.read_csv('./data/validation_data.csv')
# # less_toxic_count = ctv.transform(validation_data.less_toxic)
# # more_toxic_count = ctv.transform(validation_data.more_toxic)

# less_toxic_w2v = np.array([text.vector for text in 
#     nlp.pipe(validation_data.less_toxic, 
#         disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])])
# more_toxic_w2v = np.array([text.vector for text in 
#     nlp.pipe(validation_data.more_toxic, 
#         disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])])

# print(less_toxic_w2v.shape)
# print(more_toxic_w2v.shape)

# t = time()
# # Test our model in validation dataset
# mean_score = model_predict(less_toxic_w2v[:], more_toxic_w2v[:], xgb_regressor_w2v)
# print('Mean Score: ', mean_score)
# print('Time to validate: {} mins'.format(round((time() - t) / 60, 2)))





