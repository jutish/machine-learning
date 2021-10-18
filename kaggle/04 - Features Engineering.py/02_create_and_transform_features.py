import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs
import numpy as np

# Set plot parameters
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes',
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
)

# Load data
autos = pd.read_csv('./recursos/Automobile_data.csv')
autos = autos.replace('?',value=np.nan).dropna(axis=0)
autos[['stroke','bore']] = autos[['stroke','bore']].astype(float)
autos['num-of-cylinders'] = autos['num-of-cylinders'] \
                            .replace(['four','five','six','three', 'eight'] \
                            ,value=[4,5,6,3,8])

concrete_cols = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
                  "Superplasticizer", "CoarseAggregate", "FineAggregate","Age","CCS"]
concrete = pd.read_excel('./recursos/Concrete_Data.xls', names=concrete_cols)

customer_cols = ['Customer', 'State', 'CustomerLifetimeValue', 'Response', 'Coverage',
               'Education', 'Effective oDate', 'EmploymentStatus', 'Gender',
               'Income', 'LocationCode', 'Marital Status', 'MonthlyPremiumAuto',
               'MonthsSinceLastClaim', 'MonthsSincePolicyInception',
               'NumberOfOpenComplaints', 'NumberOfPolicies', 'PolicyType',
               'Policy', 'RenewOfferType', 'SalesChannel', 'TotalClaimAmount',
               'VehicleClass', 'VehicleSize']
customer = pd.read_csv('./recursos/Watson_Customer_review.csv', names=customer_cols, header=0)

########### Mathematical Transforms #################
# The "stroke ratio", for instance, is a measure of how efficient an engine 
# is versus how performant:
autos['stroke_ratio'] = autos.stroke / autos.bore

# The more complicated a combination is, the more difficult it will be for a 
# model to learn, like this formula for an engine's "displacement", a measure of 
# its power:
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos['num-of-cylinders']
)

################# Counts #######################
# In the Concrete dataset are the amounts of components in a concrete 
# formulation. Many formulations lack one or more components 
# (that is, the component has a value of 0). This will count how many components 
# are in a formulation with the dataframe's built-in greater-than gt method:
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete['Components'] = concrete[components].gt(0).sum(axis=1)
print(concrete[components + ['Components']])

############ Building-Up and Breaking-Down Features ################
# The str accessor lets you apply string methods like split directly to columns. 
# The Customer Lifetime Value dataset contains features describing customers of 
# an insurance company. From the Policy feature, we could separate the Type from 
# the Level of coverage:
customer[['type', 'level']] = customer['PolicyType'].str.split(" ", expand=True)

# You could also join simple features into a composed feature if you had reason 
# to believe there was some interaction in the combination:
# print(autos.columns)
autos['make_and_style'] = autos['make'] + '_' + autos['body-style']
print(autos[['make','body-style','make_and_style']])

###### Group Transforms ########
customer['AverageIncome'] = customer.groupby(['State'])['Income'].transform('mean')
average_i = customer.groupby(['State'])['Income'].mean()
print(customer[["State", "Income", "AverageIncome"]].head(10))

# The mean function is a built-in dataframe method, which means we can pass it as
# a string to transform. Other handy methods include max, min, median, var, std, 
# and count. Here's how you could calculate the frequency with which each state 
# occurs in the dataset:
customer['StateFreq'] = customer.groupby(['State'])['State'].transform('count') / customer.State.count()
print(customer[['State','StateFreq']])

# You could use a transform like this to create a "frequency encoding" for a 
# categorical feature.
# If you're using training and validation splits, to preserve their independence, 
# it's best to create a grouped feature using only the training set and then join 
# it to the validation set. We can use the validation set's merge method after 
# creating a unique set of values with drop_duplicates on the training set:
# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["TotalClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

print(df_valid[["Coverage", "AverageClaim"]].head(10))

####### Categorical and numeric interaction #########

# If you've discovered an interaction effect between a numeric feature and a 
# categorical feature, you might want to model it explicitly using a one-hot 
# encoding, like so:
# One-hot encode Categorical feature, adding a column prefix "Cat"
cars_new = pd.get_dummies(autos['fuel-type'], prefix="FT")
# # # Multiply row-by-row
cars_new = cars_new.mul(autos.price, axis=0)
print(cars_new)






# Tips on Creating Features
# It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:
# Linear models learn sums and differences naturally, but can't learn anything more complex.
# Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.
# Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
# Tree models can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.
# Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.