
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier
from resources.clean_data import clean_data
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load training data
training_df = pd.read_csv('./data/ruddit_comments_score.csv')

# Cleaning training_df
# clean_df = clean_data(training_df.body)
# training_df = training_df.iloc[clean_df.index]
# training_df['body'] = clean_df.copy()

# Make our bag of word and then split. We can do this just because is Bag of Words
y = training_df.pop('score')
X = training_df['body'].copy()

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
#     test_size=0.33, random_state=0)
# print('Train Shape: ', X_train.shape, 'Train label shape:', y_train.shape)
# print('Test Shape: ', X_test.shape, 'Test label shape:', y_test.shape)

# tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode', 
#     lowercase=True, 
#     ngram_range=(1,3),
#     stop_words='english')
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# tfidf_test = tfidf_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_tfidf, y, 
    test_size=0.1, random_state=0)
print('Train Shape: ', X_train.shape, 'Train label shape:', y_train.shape)
print('Test Shape: ', X_test.shape, 'Test label shape:', y_test.shape)

# # Train our model
# # Using XGBoost training on custom word vector
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
my_model.fit(X_train, y_train, early_stopping_rounds=5,
             eval_set=[(X_test, y_test)], verbose=False)
predictions = my_model.predict(X_test)
mae = mean_absolute_error(predictions, y_test)
print('Mean Absolute Error: ', mae)


# Validate our model
def model_score_pairs(less_toxic, more_toxic, model):
    less_score = model.predict(less_toxic)
    more_score = model.predict(more_toxic)
    return {'less_toxic':less_score[0], 'more_toxic':more_score[0]}


def model_predict(less_toxic, more_toxic, model):
    result = pd.DataFrame(columns=['less_toxic', 'more_toxic'])
    for i in range(less_toxic.shape[0]):
        model_score = model_score_pairs(less_toxic[i], more_toxic[i], model)
        result = result.append(model_score, ignore_index=True)
    result['value'] = result['less_toxic'] < result['more_toxic']
    return result.value.mean()


# Transform our validation data using our TF-IDF Instance.
validation_data = pd.read_csv('./data/validation_data.csv')
less_toxic_count = tfidf_vectorizer.transform(validation_data.less_toxic)
more_toxic_count = tfidf_vectorizer.transform(validation_data.more_toxic)

# Test our model in validation dataset
mean_score = model_predict(less_toxic_count[:], more_toxic_count[:], my_model)
print(mean_score)