from pulp import *
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample *.csv as input
families = np.arange(1,5001) #  There are 5000 Families
options = np.arange(1,12) #  They have to choise a top 10 of dates
days = np.arange(1,101) #  Available dates to pick
dates = np.array([random.sample(range(1,101),10) + [0] for _ in range(5000)])
example = [[f,o] for f in families for o in options]
df = pd.DataFrame(example, columns=['familie','top'])
df['date'] = dates.reshape(55000,)
df.to_csv('families_choises.csv',sep=';', index=False)

#***********************************************************
#************** Model Definition using PuLP ****************
#***********************************************************

# Read *.csv input
path = 'families_choises.csv'
df = pd.read_csv(path, dtype={'date':'int64'},sep=';')

# Add the cost of every choise
costs = np.array([[0,50,59,109,209,218,318,336,436,735,934] for _ in range(5000)])
df['costs'] = costs.reshape(55000)

# Add the variables name for LpProblem
df['lp_var'] = df.familie.astype('str') + '_'+ \
               df.top.astype('str') + '_' + \
               df.date.astype('str')

# Create LpVariables
# Each variable is defined as familie_top_date
# The values of variables are between 0 and Infinite and they are Integer
variables = df['lp_var'].to_numpy()
lp_variables = LpVariable.dicts('X', variables, 0, None, LpInteger)

# Set objetive: Minimize total cost of penalties
model = LpProblem("Santas_Village", LpMinimize)
grouped = df.groupby('top')
objective = []
for name, group in grouped:
    group_vars = group['lp_var'].to_numpy()
    group_value = group['costs'].max() 
    objective+=(lpSum([group_value * lp_variables[i] for i in group_vars]))

model += objective, 'Minimize total cost of penalties'

# Set Restriction 1:
# Each day we have to receive more than 125 families

# Set Restriction 2:
# Each day we have to receive less than 300 families

# Set restriction 3:
# Each family can pick just one date


# Save the model to analyze in a visual way
model.writeLP("Santas_Village.lp")

