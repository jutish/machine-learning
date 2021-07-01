import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('./recursos/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Using XGBoost
my_model = XGBRegressor(random_state=0)
my_model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = my_model.predict(X_valid)
mae = mean_absolute_error(predictions, y_valid)
print('Mean Absolute Error: ', round(mae,2))

# XGBoost parameters:

# n_estimators
# It is equal to the number of models that we include in the ensemble.
# Typical values range from 100-1000
my_model = XGBRegressor(n_estimators = 500)
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_valid)
mae = mean_absolute_error(predictions, y_valid)
print('Mean Absolute Error: ', round(mae,2))

# early_stopping_rounds
# Early stopping causes the model to stop iterating when the validation score
# stops improving. Setting early_stopping_rounds=5 will stop after 5 straight
# rounds of deteriorating validation score. When using early_stopping_rounds,
# you also need to set aside some data for calculating the validation scores
# this is done by setting the eval_set parameter.
my_model = XGBRegressor(n_estimators = 500)
my_model.fit(X_train, y_train, early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)], verbose=False)
predictions = my_model.predict(X_valid)
mae = mean_absolute_error(predictions, y_valid)
print('Mean Absolute Error: ', round(mae,2))

# learning_rate
# As default, XGBoost sets learning_rate=0.1.
# In general, a small learning rate and large number of estimators will yield
# more accurate XGBoost models, though it will also take the model longer to
# train since it does more iterations through the cycle. As default, XGBoost
# sets learning_rate=0.1.
my_model = XGBRegressor(n_estimators = 1000, learning_rate=0.05)
my_model.fit(X_train, y_train, early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)], verbose=False)
predictions = my_model.predict(X_valid)
mae = mean_absolute_error(predictions, y_valid)
print('Mean Absolute Error: ', round(mae,2))

# n_jobs
# It's common to set the parameter n_jobs equal to the number of cores on your
# machine. On smaller datasets, this won't help.
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
predictions = my_model.predict(X_valid)
mae = mean_absolute_error(predictions, y_valid)
print('Mean Absolute Error: ', round(mae,2))