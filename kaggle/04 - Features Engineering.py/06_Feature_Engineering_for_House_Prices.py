# Introduction
# Welcome to the feature engineering project for the House Prices - Advanced
# Regression Techniques competition! This competition uses nearly the same data
# you used in the exercises of the Feature Engineering course. We'll collect
# together the work you did into a complete project which you can build off of
# with ideas of your own.

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression

# Set matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes',
        labelweight='bold',
        labelsize='large',
        titleweight='bold',
        titlesize=14,
        titlepad=10)

# Data Preprocessing

# Before we can do any feature engineering, we need to preprocess the data to
# get it in a form suitable for analysis.
# We'll need to:


# Load the data from CSV files 
# Clean the data to fix any errors or inconsistencies 
# Encode the statistical data type (numeric, categorical)
# Impute any missing values 
def load_data(preprop=True):
    # Read data
    path = './recursos/house_pricing/'
    df_train = pd.read_csv(path + 'train.csv', index_col='Id')
    df_test = pd.read_csv(path + 'test.csv', index_col='Id')
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    # Prepocessing
    if preprop:
        df = clean(df)
        df = encode(df)
        df = impute(df)
    # # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return df_train, df_test


# Clean Data 
# Some of the categorical features in this dataset have what are
# apparently typos in their categories:
# df, _ = load_data(clean=False)
# print(df.Exterior2nd.unique())
# print(df.Exterior2nd.value_counts())
# print(df['MSSubClass'].astype('category').cat.categories)
# print(df.select_dtypes('number'))



# Comparing these to data_description.txt shows us what needs cleaning. We'll
# take care of a couple of issues here, but you might want to evaluate this
# data further.
def clean(df):
    # Fix a value of Exterior2nd
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Brk Cmn': 'BrkComm'})
    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built. Where function could be used to match
    # and replace. See documentation of Where.
    df['GarageYrBlt'] = df['GarageYrBlt'].where(df['GarageYrBlt'] < 2010, df['YearBuilt'])
    # Names beginning with numbers are awkward to work with
    df.rename(columns={
        '1stFlrSF': 'FirstFlrSF',
        '2ndFlrSF': 'SecondFlrSF',
        '3SsnPorch': 'Threeseasonporch'
    }, inplace=True
    )
    return df

# Encode the Statistical Data Type
# Pandas has Python types corresponding to the standard statistical types
# (numeric, categorical, etc.). Encoding each feature with its correct type
# helps ensure each feature is treated appropriately by whatever functions we
# use, and makes it easier for us to apply transformations consistently. This
# hidden cell defines the encode function:

# The numeric features are already encoded correctly (`float` for
# continuous, `int` for discrete), but the categoricals we'll need to
# do ourselves. Note in particular, that the `MSSubClass` feature is
# read as an `int` type, but is actually a (nominative) categorical.

# The nominative (unordered) categorical features
features_nom = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 
                'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
                'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 
                 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 
                 'SaleType', 'SaleCondition']
# print(df[features_nom].dtypes)

# The ordinal (ordered) categorical features
# Pandas calls the categories 'levels'
# print(df['ExterQual'].value_counts())
five_levels = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
ten_levels = list(range(10))
ordered_levels = {
    'OverallQual': ten_levels,
    'OverallCond': ten_levels,
    'ExterQual': five_levels,
    'ExterCond': five_levels,
    'BsmtQual': five_levels,
    'BsmtCond': five_levels,
    'HeatingQC': five_levels,
    'KitchenQual': five_levels,
    'FireplaceQu': five_levels,
    'GarageQual': five_levels,
    'GarageCond': five_levels,
    'PoolQC': five_levels,
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'Functional': ['Sal', 'Sev', 'Maj1', 'Maj2', 'Mod', 'Min2', 'Min1', 'Typ'],
    'GarageFinish': ['Unf', 'RFn', 'Fin'],
    'PavedDrive': ['N', 'P', 'Y'],
    'Utilities': ['NoSeWa', 'NoSewr', 'AllPub'],
    'CentralAir': ['N', 'Y'],
    'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
    'Fence': ['MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
}

# Add a None level for missing values
ordered_levels = {key: ['None'] + value for key, value in
                  ordered_levels.items()}
# print(ordered_levels)

def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype('category')
        # Add a None category for missing values
        if 'None' not in df[name].cat.categories:
            df[name].cat.add_categories('None', inplace=True)
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df


# Handle Missing Values
# Handling missing values now will make the feature engineering go more
# smoothly. We'll impute 0 for missing numeric values and 'None' for missing
# categorical values. You might like to experiment with other imputation
# strategies. In particular, you could try creating 'missing value' indicators:
# 1 whenever a value was imputed and 0 otherwise.
def impute(df):
    for name in df.select_dtypes('number'):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes('category'):
        df[name] = df[name].fillna('None')
    return df

# Load Data
# And now we can call the data loader and get the processed data splits:
df_train, df_test = load_data()
# print(df_train.Electrical.cat.codes, df_train.Electrical)
# print(df_train.info())
# print(df_test.info())


# Establish Baseline
# Finally, let's establish a baseline score to judge our feature engineering against.
# Here is the function we created in Lesson 1 that will compute the
# cross-validated RMSLE score for a feature set. We've used XGBoost for our
# model, but you might want to experiment with other models
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(model, X, log_y, cv=5, 
                            scoring='neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# X = df_train.copy()
# y = X.pop('SalePrice')
# baseline_score = score_dataset(X, y)
# print(f'Baseline_score: {baseline_score:.5f} RMSLE')


# Step 2 - Feature Utility Scores 

# In Lesson 2 we saw how to use mutual
# information to compute a utility score for a feature, giving you an
# indication of how much potential the feature has. This hidden cell defines
# the two utility functions we used, make_mi_scores and plot_mi_scores:
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object','category']):
        # X[colname] = X[colname].cat.codes  # My idea, why not?
        X[colname], _ = X[colname].factorize()  # Original code from kaggle
    # All discrete features should now have integer dyptes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, 
        discrete_features = discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=True)
    return mi_scores

def plot_mi_scores(scores):
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
    plt.show()

# Let's look at our feature scores again:
X = df_train.copy()
y = X.pop('SalePrice')
mi_scores = make_mi_scores(X, y)
# plot_mi_scores(mi_scores)
# print(mi_scores.sort_values(ascending=False))

# You can see that we have a number of features that are highly informative and
# also some that don't seem to be informative at all (at least by themselves).
# As we talked about in Tutorial 2, the top scoring features will usually
# pay-off the most during feature development, so it could be a good idea to
# focus your efforts on those. On the other hand, training on uninformative
# features can lead to overfitting. So, the features with 0.0 scores we'll drop
# entirely:
# print(mi_scores[mi_scores==0].index)
def drop_uninformative(df, mi_scores):
    X = df.copy()
    X = X.drop(mi_scores[mi_scores==0].index, axis=1)
    return X

# score = score_dataset(X, y)
# print(f'Baseline score without 0.00 MI features: {score:.5f} RMSLE')

# Step 3 - Create Features

# Now we'll start developing our feature set.
# To make our feature engineering workflow more modular, we'll define a function
# that will take a prepared dataframe and pass it through a pipeline of
# transformations to get the final feature set. It will look something like
# this:


# Let's go ahead and define one transformation now, a label encoding for the
# categorical features:
def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(['category','object']):
        X[colname] = X[colname].cat.codes

    return X

# A label encoding is okay for any kind of categorical feature when you're using a
# tree-ensemble like XGBoost, even for unordered categories. If you wanted to try
# a linear regression model (also popular in this competition), you would instead
# want to use a one-hot encoding, especially for the features with unordered
# categories.

# Create Features with Pandas 
# This cell reproduces the work you did in Exercise
# 3, where you applied strategies for creating features in Pandas. Modify or
# add to these functions to try out other feature combinations.

def mathematical_transforms(df):
    X = pd.DataFrame()
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    return X


def interactions(df):
    X = pd.get_dummies(df.BldgType, prefix='Bldg')  # Provides a one hot enconder
    X = X.mul(df.GrLivArea, axis=0)
    return X


# Count how many porch types are present in a house
def counts(df):
    X = pd.DataFrame()
    X['PorchTypes'] = df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                     'Threeseasonporch', 'ScreenPorch']].gt(0.0).sum(axis=1)
    return X


# Note: I think that this doesn't work with my data, because I have only numbers
def break_down(df):
    X = pd.DataFrame
    X['MSClass'] = df['MSSubClass'].str.split('_', n=1, expand=True)[0]
    return X

# Get mean of GrLivArea by Neighborhood
def group_transforms(df):
    X = pd.DataFrame()
    X['MedNhbArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('mean')
    return X

# Here are some ideas for other transforms you could explore:

# Interactions between the quality Qual and condition Cond features. OverallQual,
# for instance, was a high-scoring feature. You could try combining it with
# OverallCond by converting both to integer type and taking a product. Square
# roots of area features. This would convert units of square feet to just feet.
# Logarithms of numeric features. If a feature has a skewed distribution,
# applying a logarithm can help normalize it. Interactions between numeric and
# categorical features that describe the same thing. You could look at
# interactions between BsmtQual and TotalBsmtSF, for instance. Other group
# statistics in Neighboorhood. We did the median of GrLivArea. Looking at mean,
# std, or count could be interesting. You could also try combining the group
# statistics with other features. Maybe the difference of GrLivArea and the
# median is important? k-Means Clu

# k-Means Clustering

# The first unsupervised algorithm we used to create features was k-means
# clustering. We saw that you could either use the cluster labels as a feature
# (a column with 0, 1, 2, ...) or you could use the distance of the
# observations to each cluster. We saw how these features can sometimes be
# effective at untangling complicated spatial relationships.
cluster_features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]

# Since k-means clustering is sensitive to scale, it can be a good idea rescale 
# or normalize data with extreme values.
# This function returns a new Column called Cluster indicating to wich cluster
# belongs the features.
def cluster_labels(df, cluster_features, n_cluster=20):
    X = df.copy()
    X_scaled = X.loc[:, cluster_features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_cluster, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new['Cluster'] = kmeans.fit_predict(X_scaled)
    return X_new


# This function returns the distance between features and each centroid.
def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    # Label features and join to dataset
    X_cd = pd.DataFrame(X_cd, columns=[f'Centroid_{i}' for i in range(X_cd.shape[1])])
    return X_cd

# PCA Principal Component Analysis 
# PCA was the second unsupervised model we used
# for feature creation. We saw how it could be used to decompose the
# variational structure in the data. The PCA algorithm gave us loadings which
# described each component of variation, and also the components which were the
# transformed datapoints. The loadings can suggest features to create and the
# components we can use as features directly.

# Here are the utility functions from the PCA lesson: The new features PCA
# constructs are actually just linear combinations (weighted sums) of the
# original features: 
# df["PC1"] = 0.707 * X["Height"] + 0.707 * X["Diameter"] 
# df["PC2"] = 0.707 * X ["Height"] - 0.707 * X["Diameter"] 
# These new features are called the principal components of the data. 
# The weights themselves are called loadings. There will be as many principal 
# components as there are features in the original dataset: if we had used ten 
# features instead of two, we would have ended up with ten components.

def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    components_name = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=components_name)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # Transpose the matrix of loadings
        columns=components_name,
        index=X.columns
        )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    figure, ax = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n+1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel='Component', title='% Explained variance', ylim=(0.0, 1.0))
    # Cumulative variance
    cv = evr.cumsum()
    axs[1].plot(grid, cv)
    axs[1].set(xlabel='Component', title='% Cumulative variance', ylim=(0.0, 1.0))
    # Set up figure
    fig.set(figwidth=width, dpi=dpi)
    return axs

# And here are transforms that produce the features from the Exercise 5. You
# might want to change these if you came up with a different answer.

def pca_inspired(df):
    X = pd.DataFrame()
    X["Feature1"] = df.GrLivArea + df.TotalBsmtSF
    X["Feature2"] = df.YearRemodAdd * df.TotalBsmtSF
    return X


def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca

pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]

# These are only a couple ways you could use the principal components. You could
# also try clustering using one or more components. One thing to note is that PCA
# doesn't change the distance between points -- it's just like a rotation. So
# clustering with the full set of components is the same as clustering with the
# original features. Instead, pick some subset of components, maybe those with
# the most variance or the highest MI scores.

# For further analysis, you might want to look at a correlation matrix for the
# dataset:

def corrplot(df, method="pearson", annot=True, **kwargs):
    sns.clustermap(
        df.corr(method),
        vmin=-1.0,
        vmax=1.0,
        cmap="icefire",
        method="complete",
        annot=annot,
        **kwargs,
    )
    plt.show()

# corrplot(df_train, annot=None)

# PCA Application - Indicate Outliers 
# In Exercise 5, you applied PCA to
# determine houses that were outliers, that is, houses having values not well
# represented in the rest of the data. You saw that there was a group of houses
# in the Edwards neighborhood having a SaleCondition of Partial whose values
# were especially extreme.

# Some models can benefit from having these outliers indicated, which is what
# this next transform will do.

def indicate_outliers(df):
    X_new = pd.DataFrame()
    X_new['Outlier'] = (df.Neighborhood == "Edwards") & (df.SaleCondition == "Partial")
    return X_new

# Target Encoding 
# Needing a separate holdout set to create a target encoding is
# rather wasteful of data. In Tutorial 6 we used 25% of our dataset just to
# encode a single feature, Zipcode. The data from the other features in that
# 25% we didn't get to use at all.

# There is, however, a way you can use target encoding without having to use
# held-out encoding data. It's basically the same trick used in
# cross-validation:

# Split the data into folds, each fold having two splits of the dataset. Train
# the encoder on one split but transform the values of the other. Repeat for
# all the splits. This way, training and transformation always take place on
# independent sets of data, just like when you use a holdout set but without
# any data going to waste.

# In the next hidden cell is a wrapper you can use with any target encoder:
class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(X.iloc[idx_encode, :], y.iloc[idx_encode])
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + '_encoded' for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce( lambda x, y: x.add(y, fill_value=0), X_encoded_list
                            ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

# Use it like:
# You can turn any of the encoders from the category_encoders library into a
# cross-fold encoder. The CatBoostEncoder would be worth trying. It's similar to
# MEstimateEncoder but uses some tricks to better prevent overfitting. Its
# smoothing parameter is called a instead of m. 
X, _ = load_data()
encoder = CrossFoldEncoder (MEstimateEncoder, m=1) 
X_encoded = encoder.fit_transform(X, y, cols=["MSSubClass"])

# Create Final Feature Set Now let's combine everything together. Putting the
# transformations into separate functions makes it easier to experiment with
# various combinations. The ones I left uncommented I found gave the best
# results. You should experiment with you own ideas though! Modify any of these
# transformations or come up with some of your own to add to the pipeline.

def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop('SalePrice')
    mi_scores = make_mi_scores(X, y)

    # Combine splits if test data is given
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop('SalePrice')
        X = pd.concat([X, X_test])

    # Lesson 2 - Mutual Information
    X = drop_uninformative(X, mi_scores)

    # Lesson 3 - Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    X = X.join(counts(X))
    # X = X.join(break_down(X))
    X = X.join(group_transforms(X))

    # Lesson 4 - Clustering
    # X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    # X = X.join(cluster_distance(X, cluster_features, n_clusters=20))

    # Lesson 5 - PCA
    X = X.join(pca_inspired(X))
    # X = X.join(pca_components(X, pca_features))
    # X = X.join(indicate_outliers(X))

    X = label_encode(X)

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Lesson 6 - Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))

    if df_test is not None:
        return X, X_test
    else:
        return X

df_train, df_test = load_data()
score = score_dataset(df_train.drop(['SalePrice'],axis=1), df_train['SalePrice'])
print('Score Without Transforms: ', score)
X_train = create_features(df_train)
y_train = df_train.loc[:, "SalePrice"]
score = score_dataset(X_train, y_train)
print('Score With Transforms: ', score)

# Step 4 - Hyperparameter Tuning
# Step 4 - Hyperparameter Tuning At this stage, you might like to do some
# hyperparameter tuning with XGBoost before creating your final submission.

X_train = create_features(df_train)
y_train = df_train.loc[:, "SalePrice"]

xgb_params = dict(
    max_depth=6,           # maximum depth of each tree - try 2 to 10
    learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
    n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
    min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
    colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
    subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
    reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
    num_parallel_tree=1,   # set > 1 for boosted random forests
)

xgb = XGBRegressor(**xgb_params)
score = score_dataset(X_train, y_train, xgb)
print('Score With XGBoost Hyperparm: ', score)