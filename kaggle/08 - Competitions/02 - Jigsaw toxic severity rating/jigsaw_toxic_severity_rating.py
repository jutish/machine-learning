import pandas as pd
import numpy as np
import spacy
import praw
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load or make a ruddit comments score
if os.path.isfile('ruddit_comments_score.csv'):
    # Load dataset if exists
    df = pd.read_csv('ruddit_comments_score.csv',index_col='comment_id')
    print('Ruddit comments score loaded! Shape: ', df.shape)
else:
    # Download Ruddit dataset from Reddit using Praw
    ruddit = pd.read_csv('Ruddit.csv', index_col='comment_id')
    ruddit_ids = ['t1_'+comment for comment in ruddit.index]
    # Connect to reddit API using praw on read_only mode to download Ruddit comments
    # by comment_id using.
    CLIENT_ID = 'KtPrmj4MScZOqy66FrCvGw'
    CLIENT_SECRET = '79BxNg31gX-DnXouQArWSCNMrsOjBQ'
    USERAGENT = 'toxiccomments'
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USERAGENT,
    )
    # Make a dictionary with all de comments and their score
    data = {}
    for comment in reddit.info(ruddit_ids):
        idx = comment.id
        body = comment.body
        score = ruddit.loc[idx]['offensiveness_score']
        data[idx] = {'body':body,'score':score}
    df = pd.DataFrame(data).T
    # Write comments and their score in a *.csv for future analysis
    df.index.name = 'comment_id'
    df.to_csv('ruddit_comments_score.csv')
    print('Ruddit comments score saved! Shape: ',df.shape)

# Remove [deleted] comments
df = df[df.body != '[deleted]']

#######################################################################
# First approach "Cosine Similarity"                                  #
#######################################################################
# We don't train a model. The idea is take each text to score, get its Word2Vec
# using Spacy and compare with the Vectorize version of Ruddit Dataset using
# Cosine Similarity. Then use the score of the most close text of Ruddit to score
# our text.
# Get or load a vectorize version of Ruddit DataFrame
nlp = spacy.load('en_core_web_lg')
if os.path.isfile('df_vectors.npy'):
    df_vectors = np.load('df_vectors.npy')
    print('Ruddit comments vectorized loaded! Shape: ',df_vectors.shape)
else:
    with nlp.disable_pipes():
        df_vectors = np.array([nlp(text).vector for text in df.body])
        np.save('df_vectors',df_vectors)
        print('Ruddit comments vectorized saved! Shape: ',df_vectors.shape)

# Load validation data
validation_data = pd.read_csv('validation_data.csv')

# Find the most similar comment to our text.
def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

# Define score function
def score(comment):
    # Vectorize the comment to score
    comment_vec = nlp(comment).vector
    # Center the df_vectors substracting the mean for the vectors
    vec_mean = df_vectors.mean(axis=0)
    centered_vec = df_vectors - vec_mean
    # Create an array with cosine similarity
    sims = np.array([cosine_similarity(comment_vec - vec_mean, df_vec)
        for df_vec in centered_vec])
    most_similar = sims.argmax()
    return df.iloc[most_similar]['score']

def score_pairs(comments):
    less_toxic = comments[0]
    more_toxic = comments[1]
    less_score = score(less_toxic)
    more_score = score(more_toxic)
    return pd.Series([less_score, more_score])

def predict(validation_data):
    test_df = validation_data[['less_toxic','more_toxic']]
    result = test_df.apply(score_pairs, raw=True, axis=1)
    result['value'] = result['less_toxic'] < result['more_toxic']
    return result.value.mean()

# Load comments to score
# c2s = pd.read_csv('comments_to_score.csv')
# # Score and make a submission file
# output = pd.DataFrame([{'comment_id':cm.comment_id, 'score':score(cm.text)} 
#     for _,cm in c2s.iterrows()])
# output.to_csv('submission.csv', index=False)
# scores = pd.read_csv('submission.csv',index_col='comment_id')
# comments = pd.read_csv('comments_to_score.csv', index_col='comment_id')
# comments_score = scores.join(comments)
# comments_score.to_csv('comments_score.csv',sep=';')

##################################################################
# Second Approach Training a model based on Ruddit Comments Score#
##################################################################

# Split our dataset based on Ruddits dataset vecotirized
X_train, X_test, y_train, y_test = train_test_split(df_vectors, df.score,
    test_size=0.1, random_state=1)

# # # Using XGBoost
# my_model = XGBRegressor(n_estimators = 1000, learning_rate=0.05, random_state=0)
# my_model.fit(X_train, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test, y_test)], verbose=False)
# predictions = my_model.predict(X_test)
# mae = mean_absolute_error(predictions, y_test)
# print('Mean Absolute Error: ', mae)

my_model = XGBRegressor(random_state=0)
my_model.fit(X_train, y_train)
print(X_train.shape)

def model_score_pairs(comments):
    less_toxic = comments[0]
    more_toxic = comments[1]
    less_vector = nlp(less_toxic).vector.reshape(1,300)
    more_vector = nlp(more_toxic).vector.reshape(1,300)
    less_score = my_model.predict(less_vector)
    more_score = my_model.predict(more_vector)
    return pd.Series([less_score, more_score])
    # return pd.Series([1,2])

def model_predict(validation_data):
    test_df = validation_data[['less_toxic','more_toxic']]
    result = test_df.apply(model_score_pairs, raw=True, axis=1)
    result['value'] = result['less_toxic'] < result['more_toxic']
    # print(result)
    return result.value.mean()

val_data_test = validation_data.iloc[:]
mean_score = model_predict(val_data_test)
print(mean_score)