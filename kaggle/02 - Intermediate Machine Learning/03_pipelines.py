import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('./recursos/melb_data.csv')

# Remove rows with missing targets
data = data.dropna(subset=['Price'], axis=0)

# Separate predictors from target
X = data.drop(['Price'], axis=1)
y = data.Price

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0,
                                                      train_size=0.8)

# Cardinality means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but
# arbitrary)
cat_cols = [col for col in X_train.columns if (X_train[col].dtype=='object') and 
                                              (X_train[col].nunique() < 10)]
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 
                                                                     'float64']]

# Keep selected columns only
my_cols = cat_cols + num_cols
X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()

# We define the pipline in three steps

# Step 1: Define Preprocessing Steps
# imputes missing values in numerical data, and
# imputes missing values and applies a one-hot encoding to categorical data.

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
    ('num', numerical_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)])

# Step 2: Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Step 3: Create and evaluate the pipeline
# With the pipeline, we preprocess the training data and fit the model in a
# single line of code. (In contrast, without a pipeline, we have to do
# imputation, one-hot encoding, and model training in separate steps.
# This becomes especially messy if we have to deal with both numerical and
# categorical variables!)
# With the pipeline, we supply the unprocessed features in X_valid to the
# predict() command, and the pipeline automatically preprocesses the features
# before generating predictions. (However, without a pipeline, we have to
# remember to preprocess the validation data before making predictions.)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                              ('model', model)])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE: ', score)
