import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv('./recursos/melb_data.csv')
print('\nShape of data: ', df.shape)

# Cheq for Null values
print('\nColumns with Null values: \n', df.isnull().sum())

# Drop rows with NULL prices
df = df.dropna(subset=['Price'], axis=0)

# Drop columns with Null values (simplest approach)
df = df.dropna(axis=1)
print('\nShape of data after drops: ', df.shape)

# Get parameters and output
X = df.drop(['Price'], axis=1)
y = df.Price

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [col for col in X.columns if X[col].nunique() < 10
                        and X[col].dtype == 'object']

# Select numerical columns
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Keep selected columns only
columns = low_cardinality_cols + numerical_cols
X = X[columns].copy()

# Split intro train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, 
                                                     random_state=0)

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

# Function for compare differents approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(random_state=100, n_estimators=100)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    return mae

# Score from Approach 1 (Drop Categorical Variables)
# We drop the object columns with the select_dtypes() method.
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Score from Approach 2 (Label Encoding)
# Scikit-learn has a LabelEncoder class that can be used to get label encodings.
# We loop over the categorical variables and apply the label encoder separately
# to each column.
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# Score from Approach 3 (One-Hot Encoding)
# We use the OneHotEncoder class from scikit-learn to get one-hot encodings.
# There are a number of parameters that can be used to customize its behavior.
# Note: There is a function called "df.get_dummies()" which do the same.

# We set handle_unknown='ignore' to avoid errors when the validation data contains
# classes that aren't represented in the training data, and
# setting sparse=False ensures that the encoded columns are returned as a 
# numpy array (instead of a sparse matrix).
# To use the encoder, we supply only the categorical columns that we want to be 
# one-hot encoded. For instance, to encode the training data, we supply 
# X_train[object_cols]. (object_cols in the code cell below is a list of the 
#column names with categorical data, and so X_train[object_cols] contains 
#all of the categorical data in the training set.)

# Apply one-hot encoder to each column with categorical data
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[object_cols]))
oh_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[object_cols]))

# # One-hot encoding removed index; put it back
oh_cols_train.index = X_train.index
oh_cols_valid.index = X_valid.index

# # Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# # Add one-hot encoded columns to numerical features
oh_X_train = pd.concat([num_X_train, oh_cols_train], axis=1)
oh_X_valid = pd.concat([num_X_valid, oh_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(oh_X_train, oh_X_valid, y_train, y_valid))