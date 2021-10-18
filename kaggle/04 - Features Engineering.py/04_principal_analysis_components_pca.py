# PCA Best Practices There are a few things to keep in mind when applying PCA: PCA
# only works with numeric features, like continuous quantities or counts. PCA is
# sensitive to scale. It's good practice to standardize your data before applying
# PCA, unless you know you have good reason not to. Consider removing or
# constraining outliers, since they can an have an undue influence on the
# results.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

# Set plot parameters
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,)


# Define plot variance
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n+1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    print('evr:', evr.cumsum())
    axs[0].bar(grid, evr)
    axs[0].set(xlabel='Component',
        title='% Explained variance',
        ylim=(0.0, 1.0))
    # Cumulative variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))
    # Set up figure
    fig.set(figwidth=width, dpi=dpi)
    plt.show()


# Define make_mi_score
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# Clean data
def clean_data(cars):
    X = cars.copy().drop(['normalized-losses'],axis=1) 
    X = X.replace(to_replace='?',value=np.nan).dropna(axis=0) # Drop rows with NaN values
    y = X.pop('price').astype('int')
    cars = pd.concat([X,y], axis=1)
    cars['horsepower'] = cars['horsepower'].astype(int)
    return cars

# Read data
cars = pd.read_csv('./recursos/Automobile_data.csv')
cars = clean_data(cars)
# print(cars.head())

# We've selected four features that cover a range of properties. Each of these
# features also has a high MI score with the target, price. We'll standardize
# the data since these features aren't naturally on the same scale.
features = ["highway-mpg", "engine-size", "horsepower", "curb-weight"]
X = cars.copy()
y = X.pop('price')
X = X.loc[:,features]
# Standarize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# Now we can fit scikit-learn's PCA estimator and create the principal
# components. You can see here the first few rows of the transformed dataset.
# Create PCA
pca = PCA()
X_pca  = pca.fit_transform(X_scaled)
# Convert to data frame
components_name = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=components_name)

# After fitting, the PCA instance contains the loadings in its components_
# attribute. (Terminology for PCA is inconsistent, unfortunately. We're
# following the convention that calls the transformed columns in X_pca the
# components, which otherwise don't have a name.) We'll wrap the loadings up in
# a dataframe.
loadings = pd.DataFrame(pca.components_.T,
    columns=components_name,
    index=X.columns)
print(loadings)

# Plot explained variation
plot_variance(pca)

# Let's also look at the MI scores of the components. Not surprisingly, PC1 is
# highly informative, though the remaining components, despite their small
# variance, still have a significant relationship with price. Examining those
# components could be worthwhile to find relationships not captured by the main
# Luxury/Economy axis.
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
# print(mi_scores)

# The third component shows a contrast between horsepower and curb_weight --
# sports cars vs. wagons, it seems.
# Show dataframe sorted by PC3
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body-style", "horsepower", "curb-weight"]
# print(cars.loc[idx, cols])
