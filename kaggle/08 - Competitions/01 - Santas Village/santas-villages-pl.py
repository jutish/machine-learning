"""
Kaggle Competition: https://www.kaggle.com/c/santa-workshop-tour-2019/overview
Author: Marcelloni, Esteban
"""

from pulp import *
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a seed
random.seed(45)

# Define a few variables to create the input example
n_families = 1000
n_days = 100 # Min would be 10

# Create a sample *.csv as input
families = np.arange(1, n_families + 1) #  There are 5000 Families
options = np.arange(1,12) #  They have to choise a top 10 of dates + a dummy day
days = np.arange(1, 10 + 1) #  Available dates to pick
# Each family pick their top 10 dates (randomly in this case)
dates = np.array([random.sample(range(1, n_days + 1), 10) + [0] for _ in range(n_families)])
example = [[f,o] for f in families for o in options]
df = pd.DataFrame(example, columns=['family','top'])
df['date'] = dates.reshape(n_families * len(options))
df.to_csv('families_choises.csv',sep=';', index=False)
# plt.hist(df.date)
# plt.show()
# print((df.date.value_counts() > 125).sum())

#***********************************************************
#************** Model Definition using PuLP ****************
#***********************************************************

# Read *.csv input
path = 'families_choises.csv'
df = pd.read_csv(path, dtype={'date':'int64'},sep=';')

# Add the cost of every choise
costs = np.array([[1,50,59,109,209,218,318,336,436,735,934] for _ in range(n_families)])
df['costs'] = costs.reshape(n_families * len(options))

# Add the variables name for LpProblem
df['lp_var'] = df.family.astype('str') + '_'+ \
               df.top.astype('str') + '_' + \
               df.date.astype('str')

# Create LpVariables
# Each variable is defined as familie_top_date
variables = df['lp_var'].to_numpy()
lp_variables = LpVariable.dicts('X', variables, lowBound=0, upBound=1, cat=LpInteger)
# lp_variables = LpVariable.dicts('X', variables, cat='Binary')

# Set objetive: Minimize total cost of penaltie
model = LpProblem("Santas_Village", LpMinimize)
grouped = df.groupby('top')
objective = []
for name, group in grouped:
    group_vars = group['lp_var'].to_numpy()
    group_value = group['costs'].max() 
    objective+=(lpSum([group_value * lp_variables[i] for i in group_vars]))
model += objective, 'Minimize total cost of penalties'


# # # Set Restriction 1:
# # # Each day we have to receive more than 125 families
# # grouped = df.groupby('date')
# # for name, group in grouped:
# #     group_vars = group['lp_var'].to_numpy()
# #     model+=(lpSum([lp_variables[i] for i in group_vars]) >= 50, 
# #         f'Min Families receive on day {name}')

# # Set Restriction 2:
# # Each day we have to receive less than 300 families
# # grouped = df.groupby('date')
# # for name, group in grouped:
# #     group_vars = group['lp_var'].to_numpy()
# #     model+=(lpSum([lp_variables[i] for i in group_vars]) <= 300, 
# #         f'Max Families receive on day {name}')

# # Set restriction 3:
# # Each family receive just one date of the ten picked
grouped = df.groupby('family')
for name, group in grouped:
    group_vars = group['lp_var'].to_numpy()
    model+=(lpSum([lp_variables[i] for i in group_vars]) == 1, 
        f'Family {name} is selected for just one day ')

# Set Restriction 4:
# Each day, if we receive families, they have to be between 125 and 300.
min_fam_day = 50
max_fam_day = 300
fl = {i: LpVariable(name=f"b1_{i}", cat="Binary") for i in range(1, n_days + 1)}
grouped = df.groupby('date')
for name, group in grouped:
    if name!=0: #  If the day is <> 0 we put the restrictions
        group_vars = group['lp_var'].to_numpy()
        model+=(lpSum([lp_variables[i] for i in group_vars]) >= fl[name] * min_fam_day, 
            f'Min Families receive on day {name}')
        model+=(lpSum([lp_variables[i] for i in group_vars]) <= fl[name] * max_fam_day, 
        f'Max Families receive on day {name}')

# Set Restriction 5:
# All the flags (fl variable) have to sum less or equal to 100 (days)
model += (lpSum([f for f in fl.values()]) <= n_days, f'Flags would sum max to {n_days}')

# Set Restriction 6:
# All the flags (fl variable) have to sum more or equal to 1
model += (lpSum([f for f in fl.values()]) >= 1, f'Flags would sum min to 1')

# # Save the model to analyze in a visual way
model.writeLP("Santas_Village.lp")

# # # The problem is solved using PuLP's choice of Solver
model.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])
solution_df = pd.DataFrame(columns=['family','top','day', 'qty'])
for v in model.variables():
    if v.varValue!=0 and v.name[0]=='X':
        data = v.name.split('_')
        family = data[1]
        top = data[2]
        day = data[3]
        qty = v.varValue
        row = {'family':family, 'top':top, 'day':day, 'qty': qty}
        solution_df = solution_df.append(row, ignore_index=True)

solution_df.set_index(['day','family'], drop=True) \
    .sort_index() \
    .to_excel('Family_Schedule_Solution.xls') 



