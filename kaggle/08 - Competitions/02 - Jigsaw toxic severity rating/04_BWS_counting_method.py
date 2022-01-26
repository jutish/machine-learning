import pandas as pd
import numpy as np
import spacy
import praw
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# # Read validation dataset
# df = pd.read_csv('./data/validation_data.csv')

# # Set an id for the comments and the proportion to left and the proportion to right
# df['less_id'] = df['less_toxic'].apply(hash)
# df['less_count'] = df.groupby(['less_toxic'])['less_toxic'].transform(lambda x: len(x))

# df['more_id'] = df['more_toxic'].apply(hash)
# df['more_count'] = df.groupby(['more_toxic'])['more_toxic'].transform(lambda x: len(x))

# left = df.drop_duplicates(subset=['less_id'])[['less_id','less_toxic','less_count']]
# rigth = df.drop_duplicates(subset=['more_id'])[['more_id','more_toxic','more_count']]
# print(left.shape)
# print(rigth.shape)
# merged_df = left.merge(rigth, left_on='less_id', right_on='more_id', how='outer')
# merged_df = merged_df.fillna(0)

# merged_df['total_count'] = merged_df['less_count'] + merged_df['more_count']
# merged_df['total_score'] = merged_df['more_count']/merged_df['total_count'] -\
#                           merged_df['less_count']/merged_df['total_count']

# # Remove comments wich less than 3 reviews
# merged_df = merged_df[merged_df.total_count >=3]

# # Create a final dataset
# final_df = pd.DataFrame()
# final_df = merged_df[merged_df['less_id'] > 0][['less_id','less_toxic','total_score']]
# final_df.columns = ['more_id','more_toxic','total_score']
# final_df = final_df.append(merged_df[merged_df['more_id'] > 0][['more_id','more_toxic','total_score']], ignore_index=True)
# final_df.columns = ['comment_id','comment','score']
# final_df = final_df.drop_duplicates(subset=['comment_id'])

# # Save the dataset
# final_df.to_csv('./data/validation_data_scored.csv', index=False)

#####################################################
# Train a model based on validation_data_scored #####
# Validate against ruddit_comments_score ############
#####################################################

# df = pd.read_csv('./data/validation_data_scored.csv')
# validation_df = pd.read_csv('./data/ruddit_comments_score.csv')

# Get or load a vectorized version of training_df
# nlp = spacy.load('en_core_web_lg')
# if os.path.isfile('./data/validation_scored_vectors.npy'):
#     df_vectors = np.load('./data/validation_scored_vectors.npy')
#     print('Validation scored vectorized loaded! Shape: ', df_vectors.shape)
# else:
#     with nlp.disable_pipes():
#         df_vectors = np.array([nlp(text).vector for text in df.comment])
#         np.save('./data/validation_scored_vectors.npy', df_vectors)
#         print('Validation scored vectorized saved! Shape: ', df_vectors.shape)

# # Split our dataset based on Ruddits dataset vectorized
# X_train, X_test, y_train, y_test = train_test_split(df_vectors, df.score,
#                                                     test_size=0.1,
#                                                     random_state=1)

# # # Using XGBoost
# my_model = XGBRegressor(random_state=0, n_estimators=500)
# my_model.fit(X_train, y_train)
# predictions = my_model.predict(X_test)
# mae = mean_absolute_error(predictions, y_test)
# print('Mean Absolute Error: ', mae)

# # Using cross_val_score
# baseline = RandomForestRegressor(criterion='mae', random_state=8)
# baseline_score = cross_val_score(baseline, df_vectors, df.score, cv=5, scoring='neg_mean_absolute_error')
# print(-1 * baseline_score.mean())

#
# Train a model based on validation_data_scored + ruddit_comments_score
#