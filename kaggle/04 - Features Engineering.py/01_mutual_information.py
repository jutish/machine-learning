import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# # Load Dataset
# colnames = ['Cement','BlastFurnaceSlag','FlyAsh','Water','Superplasticizer',
#             'CoarseAgg','FineAgg','Age','CompressiveStrength']
# data = pd.read_excel('./recursos/Concrete_Data.xls', names=colnames)
# print(data.head())

# # Make a model baseline
# X = data.copy()
# y = X.pop('CompressiveStrength')
# print(X.head())
# baseline = RandomForestRegressor(criterion='mae', random_state=8)
# baseline_score = cross_val_score(baseline, X, y, cv=5, scoring='neg_mean_absolute_error')
# baseline_score = -1 * baseline_score.mean()
# print(f"MAE Baseline Score: {baseline_score:.4}")

# # Creta three new ratio features to the dataset
# # Create synthetic features
# X["FCRatio"] = X["FineAgg"] / X["CoarseAgg"]
# X["AggCmtRatio"] = (X["CoarseAgg"] + X["FineAgg"]) / X["Cement"]
# X["WtrCmtRatio"] = X["Water"] / X["Cement"]
# # Train and score model on dataset with additional ratio features
# model = RandomForestRegressor(criterion='mae', random_state=0)
# score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
# score = score.mean() * -1
# print(f"MAE Score with Ratio Features: {score:.4}")

########## Mutual Information #########

# Using Mutual Information (MI) to measure the relationship between a future
# and a target. MI 
# Mutual information describes relationships in terms of uncertainty. 
# The mutual information (MI) between two quantities is a measure of the extent 
# to which knowledge of one quantity reduces uncertainty about the other. 
# If you knew the value of a feature, how much more confident would you be 
# about the target?
# More in  https://www.kaggle.com/ryanholbrook/mutual-information

# The Automobile dataset consists of 193 cars from the 1985 model year. 
# The goal for this dataset is to predict a car's price (the target) from 23 of 
# the car's features, such as make, body_style, and horsepower. In this example, 
# we'll rank the features with mutual information and investigate the results by 
# data visualization.

# Load Data
cars = pd.read_csv('./recursos/Automobile_data.csv')

# The scikit-learn algorithm for MI treats discrete features differently from 
# continuous features. Consequently, you need to tell it which are which. 
# As a rule of thumb, anything that must have a float dtype is not discrete. 
# Categoricals (object or categorial dtype) can be treated as discrete by giving 
# them a label encoding.
X = cars.copy().drop(['normalized-losses'],axis=1) 
X = X.replace(to_replace='?',value=np.nan).dropna(axis=0) # Drop rows with NaN values
y = X.pop('price').astype('int')
cars = pd.concat([X,y], axis=1)
cars['horsepower'] = cars['horsepower'].astype(int)
# print(cars)
# Label enconding for categorical variables
for colname in X.select_dtypes('object'):
    if X[colname].nunique() < 10:
        X[colname], _ = X[colname].factorize()
    else:
        try:
            X[colname] = X[colname].astype(int)
        except:
            X[colname], _ = X[colname].factorize()
# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == 'int64'

# Scikit-learn has two mutual information metrics in its feature_selection 
# module: one for real-valued targets (mutual_info_regression) and one for 
# categorical targets (mutual_info_classif). Our target, price, is real-valued. 
# The next cell computes the MI scores for our features and wraps them up in a 
# nice dataframe.
from sklearn.feature_selection import mutual_info_regression
def make_mi_score(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Score')
    plt.show()

# Calculate and plot MI scores
mi_scores = make_mi_score(X, y, discrete_features)
plot_mi_scores(mi_scores)

# Plot scatter relation between 'curb-weight' and 'price'
sbn.relplot(x='curb-weight', y='price', data=cars)
plt.title("Relationship between 'curb-weight' and 'price'")
plt.show()
# Plot lmplot between 'horsepower' and 'price' using hue='fuel-type'
sbn.lmplot(x='horsepower', y='price', hue='fuel-type', data=cars)
plt.title("Relationship between 'horsepower' and 'price'")
plt.show()

