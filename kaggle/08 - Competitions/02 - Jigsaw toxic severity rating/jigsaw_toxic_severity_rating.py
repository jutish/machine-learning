import pandas as pd
import numpy as np
import spacy
import praw
import os

# Load or make a ruddit comments score
if os.path.isfile('ruddit_comments_score.csv'):
    # Load dataset if exists
    df = pd.read_csv('ruddit_comments_score.csv',index_col='comment_id')
    print('Ruddit comments score loaded! Shape: ', df.shape)
else:
    # Load Ruddit dataset
    ruddit = pd.read_csv('Ruddit.csv', index_col='comment_id')
    ruddit_ids = ['t1_'+comment for comment in ruddit.index]
    # Connect to reddit API using praw on read_only mode to download Ruddit comments
    # by comment_id using.
    # Note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
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




