# Most of the techniques we've seen in this course have been for numerical
# features. The technique we'll look at in this lesson, target encoding, is
# instead meant for categorical features. It's a method of encoding categories
# as numbers, like one-hot or label encoding, with the difference that it also
# uses the target to create the encoding. This makes it what we call a
# supervised feature engineering technique.
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
df = pd.read_csv('./recursos/Automobile_data.csv')
X = df.copy().drop(['normalized-losses'],axis=1) 
X = X.replace(to_replace='?',value=np.nan).dropna(axis=0) # Drop rows with NaN values
y = X.pop('price').astype('int')
cars = pd.concat([X,y], axis=1)
df = cars.copy()
cars['horsepower'] = cars['horsepower'].astype(int)
# print(cars.head())

# Target Encoding 

# Note: First you'll need to choose which features you want to
# apply a target encoding to. 
# Categorical features with a large number of
# categories are often good candidates. And if this feature has rare categories 
# inside better. Rare is for instance when one categorie occurs only a few times

# Note 2: Very important!!! The lesson is that when using a target encoder it's
# very important to use separate data sets for training the encoder and
# training the model. Otherwise the results can be very disappointing!
# See: # 3) Overfitting with Target Encoders 
# https://www.kaggle.com/estebanmarcelloni/exercise-target-encoding/edit

# A target encoding is any kind of encoding that replaces a feature's categories with
# some number derived from the target. A simple and effective version is to
# apply a group aggregation from Lesson 3, like the mean. Using the Automobiles
# dataset, this computes the average price of each vehicle's make: This kind of
# target encoding is sometimes called a mean encoding. Applied to a binary
# target, it's also called bin counting.(Other names you might come across
# include: likelihood encoding, impact encoding, and leave-one-out encoding.)

# Look at categorical variables
print('\nCategorical features:\n',cars.select_dtypes(['object']).nunique().sort_values(ascending=False))

# For our example we could use bore or stroke also, but we'll use 'make'
# Examine values of make
print('\nMake value counts:\n',cars.make.value_counts())

# Start target enconding using a simple mean of the target
cars['make_encoded'] = cars.groupby('make').price.transform('mean').round(2)
print(cars[['make','make_encoded','price']])

# Smoothing Target Encoding 
# The idea is to blend the in-category average with the overall average. Rare
# categories get less weight on their category average, while missing
# categories just get the overall average.
# See: https://www.kaggle.com/ryanholbrook/target-encoding

# In pseudocode:
# encoding = weight * in_category_mean + (1 - weight) * overall_mean
# where weight is a value between 0 and 1 calculated from the category frequency.
# An easy way to determine the value for weight is to compute an m-estimate:
# weight = n / (n + m) where n is the total number of times that category occurs
# in the all data. The parameter m determines the "smoothing factor". Larger values
# of m put more weight on the overall estimate.

m = 5  # Low values of 'm' take more account the price of the group
       # High values of 'm' take more account the price of overall. When
       # choosing a value for m, consider how noisy you expect the categories
       # to be. Does the price of a vehicle vary a great deal within each make?
       # Would you need a lot of data to get good estimates? If so, it could be
       # better to choose a larger value for m; if the average price for each
       # make were relatively stable, a smaller value could be okay.
cars['overall_price_mean'] = cars.price.mean(axis=0).round(2)  # Mean of all prices
cars['make_group_count'] = cars.groupby('make').price.transform('count')  
cars['make_group_mean'] = cars.groupby('make').price.transform('mean').round(2)
cars['weight'] = cars.make_group_count / (cars.make_group_count + m)
cars['make_smooth_encoded'] = cars.weight * cars.make_group_mean + (1 - cars.weight ) * cars.overall_price_mean 
print(cars[['make','make_group_count','make_group_mean','price','overall_price_mean','weight','make_smooth_encoded']])

# The category_encoders package in scikit-learn-contrib implements an m-estimate
# encoder, which we'll use to encode our 'make' feature.
from category_encoders  import MEstimateEncoder
# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=['make'], m=5.0)
# Fit the enconder
X = df.copy()
y = X.pop('price')
X_encoded = encoder.fit_transform(X, y)
print(X_encoded)  #if you compare 'make' feature with 'make_smooth_encoded' it's the same.


# Use Cases for Target Encoding
# Target encoding is great for:
# High-cardinality features: A feature with a large number of categories can be
# troublesome to encode: a one-hot encoding would generate too many features
# and alternatives, like a label encoding, might not be appropriate for that
# feature. A target encoding derives numbers for the categories using the
# feature's most important property: its relationship with the target.
# Domain-motivated features: From prior experience, you might suspect that a
# categorical feature should be important even if it scored poorly with a
# feature metric. A target encoding can help reveal a feature's true
# informativeness.