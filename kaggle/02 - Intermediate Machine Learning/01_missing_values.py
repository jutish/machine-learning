# Que hacer con las columnas que tienen valores nulos
# Eliminar la columna, hay casos donde funciona
# Completar la columna con el promedio de la misma "Imputation"
# Completar la columna con el promedio de la misma y agregar al dataset
# otra columna indicando donde se completaron los datos faltantes.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('./recursos/melb_data.csv')

# To keep things simple, we'll use only numerics predictors
predictors = df.drop(['Price'], axis=1)
X = predictors.select_dtypes(exclude=['object'])
y = df.Price

# Split into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      random_state=0,
                                                      train_size=0.8,
                                                      test_size=0.2)

# Define a function to compare different approaches
# This function returns the mean absolute error (MAE) from a random forest
def score_dataset(X_train, X_valid, y_train, y_valid):
    random_forest_model = RandomForestRegressor(n_estimators=10,  # Con 100 va mejor
                                                random_state=0)
    random_forest_model.fit(X_train, y_train)
    random_forest_predict = random_forest_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, random_forest_predict)
    return mae

# Approach 1 - Drop columns with missing values
# Get names of columns with at least one missing value
# Esteban Note: I could use X_train.dropna(axis=1) instead of getting columns
#                 and drop that columns
cols_with_miss = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_miss, axis=1)
reduced_X_valid = X_valid.drop(cols_with_miss, axis=1)
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# Approach 2 - Imputation
# Fill NaN with average of the column using SimpleImputer()
# Esteban Note: We use X_train to get the AVG (.fit_transform) and then
# use this AVG to transform X_valid (just .transform()) 
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# Approach 3 - Imputation Extended
# We impute the missing values, while also keeping track of which values
# were imputed
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
# Make new columns indicating what will be imputed
for col in cols_with_miss:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# Shape of training data
print('\nShape of training data: ', X_train.shape)
print('Null values per column: ')
print(X_train[cols_with_miss].isnull().sum())