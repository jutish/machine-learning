import pandas as pd
import numpy as np
import spacy
import praw
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load or make a ruddit comments score
if os.path.isfile('./data/ruddit_comments_score.csv'):
    # Load dataset if exists
    df = pd.read_csv('./data/ruddit_comments_score.csv', index_col='comment_id')
    print('Ruddit comments score loaded! Shape: ', df.shape)
else:
    # Download Ruddit dataset from Reddit using Praw
    ruddit = pd.read_csv('./data/Ruddit.csv', index_col='comment_id')
    ruddit_ids = ['t1_'+comment for comment in ruddit.index]
    # Connect to reddit API using praw on read_only mode to download
    # Ruddit comments by comment_id using.
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
        data[idx] = {'body': body, 'score': score}
    df = pd.DataFrame(data).T
    # Write comments and their score in a *.csv for future analysis
    df.index.name = 'comment_id'
    df.to_csv('./data/ruddit_comments_score.csv')
    print('Ruddit comments score saved! Shape: ', df.shape)

# Remove [deleted] comments
df = df[df.body != '[deleted]']

# Get or load a vectorize version of Ruddit DataFrame
nlp = spacy.load('en_core_web_lg')
if os.path.isfile('./data/df_vectors.npy'):
    df_vectors = np.load('./data/df_vectors.npy')
    print('Ruddit comments vectorized loaded! Shape: ', df_vectors.shape)
else:
    with nlp.disable_pipes():
        df_vectors = np.array([nlp(text).vector for text in df.body])
        np.save('./data/df_vectors', df_vectors)
        print('Ruddit comments vectorized saved! Shape: ', df_vectors.shape)

# Load validation data
validation_data = pd.read_csv('./data/validation_data.csv')

# Split our dataset based on Ruddits dataset vectorized
X_train, X_test, y_train, y_test = train_test_split(df_vectors, df.score,
                                                    test_size=0.1,
                                                    random_state=1)

# # Using XGBoost
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
my_model.fit(X_train, y_train, early_stopping_rounds=5,
             eval_set=[(X_test, y_test)], verbose=False)
predictions = my_model.predict(X_test)
mae = mean_absolute_error(predictions, y_test)
print('Mean Absolute Error: ', mae)


def model_score_pairs(comments):
    less_toxic = comments[0]
    more_toxic = comments[1]
    less_vector = nlp(less_toxic).vector.reshape(1, 300)
    more_vector = nlp(more_toxic).vector.reshape(1, 300)
    less_score = my_model.predict(less_vector)
    more_score = my_model.predict(more_vector)
    return pd.Series([less_score, more_score])
    # return pd.Series([1,2])


def model_predict(validation_data):
    test_df = validation_data[['less_toxic', 'more_toxic']]
    result = test_df.apply(model_score_pairs, raw=True, axis=1)
    result['value'] = result['less_toxic'] < result['more_toxic']
    # print(result)
    return result.value.mean()


val_data_test = validation_data.iloc[:]
mean_score = model_predict(val_data_test)
print(mean_score)
