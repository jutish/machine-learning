# Source: https://towardsdatascience.com/helping-santa-plan-with-mixed-integer-programming-mip-1951386a6ba5

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
import random


# Load data
# df = pd.read_csv('family_data.csv')
# # Check family sizes
# families_sizes = df['n_people'].value_counts().sort_index()
# plt.figure(figsize=(14,6))
# ax = sns.barplot(x = families_sizes.index, y = families_sizes)
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.0f}\n {p.get_height()/families_sizes.sum()*100:.2f}%',
#         xy=(p.get_x() + p.get_width()/2., p.get_height()), ha='center',
#         xytext=(0,5), textcoords='offset points')
# ax.set_ylim(0, 1.1*families_sizes.max())
# plt.xlabel('Family Size')
# plt.ylabel('Count')
# plt.title('Family Size Distribution')
# plt.show()

# # Look at the days provided as first choice
# first_choise = df['choice_0'].value_counts().sort_index()
# ax = sns.barplot(x=first_choise.index, y=first_choise)
# plt.xlabel('Preferred Day (Note that Day 1 == The Day Before Christmas)', 
#     fontsize=14)
# plt.annotate('Everyone wants to visit right before Christmas!!', 
#              (1, 350), 
#              textcoords ='offset points',
#              xytext =(100,0), 
#                  ha='left',
#                  va='center', 
#                  fontsize=14, 
#                  color='black',
#                  arrowprops=dict(arrowstyle='->', 
#                                  color='black'))
# plt.xticks(rotation=90)
# plt.show()

# Heatmap of family size and assigned choise
def get_cost_by_choice(num_members):
    cost_by_choice = {}
    cost_by_choice[1] = 50
    cost_by_choice[2] = 50 + 9 * num_members
    cost_by_choice[3] = 100 + 9 * num_members
    cost_by_choice[4] = 200 + 9 * num_members
    cost_by_choice[5] = 200 + 18 * num_members
    cost_by_choice[6] = 300 + 18 * num_members
    cost_by_choice[7] = 400 + 36 * num_members
    cost_by_choice[8] = 500 + (36 + 199) * num_members
    cost_by_choice[9] = 500 + (36 + 398) * num_members
    return list(cost_by_choice.values())
# matrix = [[0] + get_cost_by_choice(i) for i in range(2,9)]
# heatmap_df = pd.DataFrame(matrix, columns=list(range(1,11)), 
#     index=list(range(2,9)))
# heatmap_df.columns.name = 'Choise'
# heatmap_df.index.name =' Family Size'
# # color map
# cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
# sns.heatmap(heatmap_df.T, annot=True, fmt='g', cmap=cmap, 
#     linewidth=5, cbar_kws={"shrink": .8})
# plt.show()

# Heatmap of Accounting Cost
def calculate_daily_accounting_cost(o_previous, o_current):
    exponent = 0.5 + abs(o_current - o_previous)/50.0
    multiplier = (o_current - 125.0) / 400.0
    return multiplier * np.power(o_current, exponent)
# days = list(range(125,301))
# days_plus1 = list(range(125,301))
# matrix_df = pd.DataFrame(columns=days_plus1, index=days)
# matrix_df.columns.name = 'Previous Day'
# matrix_df.index.name = 'Current Day'
# for prev_day in matrix_df.columns:
#     matrix_df[prev_day] = [calculate_daily_accounting_cost(prev_day, curr_day) 
#     for curr_day in matrix_df.index]

# sns.heatmap(matrix_df, vmax = 500, cmap=sns.light_palette("blue", n_colors=1000))
# plt.show()

np.random.seed(42)



# Read in some data
data = pd.read_csv('family_data.csv', index_col = 0)
NUM_FAMILY = len(data)
NUM_FAMILY = 500
NUM_CHOICES = 5
NUM_DAY = 100
MIN_OCCUPANCY, MAX_OCUPPANCY = 125, 300
NUM_OCCUPANCY = MAX_OCUPPANCY - MIN_OCCUPANCY + 1
COUNT = np.arange(MIN_OCCUPANCY, MAX_OCUPPANCY + 1)

# Our initial current best solution
initial = random.sample(range(125,301),100)
print(initial)

family_size_dict = data[['n_people']].to_dict()['n_people']
cols = [f'choice_{i}' for i in range(NUM_CHOICES)]
choices = data[cols].values

# Initialize a 5000x5 matrix with the preference costs for the families
# PREF_COSTS[f][i] contains the preference cost if the f-th family
# is assigned their i-th choice
PREF_COSTS = np.zeros((NUM_FAMILY,NUM_CHOICES))
for f in range(NUM_FAMILY):
    n = family_size_dict[f]
    PREF_COSTS[f] = get_cost_by_choice(n)[:NUM_CHOICES]

# Initialize a 176x176 matrix for the accounting costs.
# ACC_COST[k][l] contains the accounting cost if the occupancy
# of day i == k and that of day i + 1 == l
ACC_COSTS = np.zeros((NUM_OCCUPANCY, NUM_OCCUPANCY))
for i in range(125, 301):
    for j in range(125, 301):
        constant = (i - 125) / 400
        diff = abs(i - j)
        ACC_COSTS[i - 125, j - 125] = constant * i ** (.5 + diff / 50)
plt.show()


def print_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        solution = []
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        for f in range(NUM_FAMILY):
            for c in range(NUM_CHOICES):
                if model.cbGetSolution(model.getVarByName('fam_{}_choice_{}'.format(f, c))):
                    solution.append(choices[f][c])
                    break
        print(solution)
        pickle.dump(solution, open('mip_{}.p'.format(int(obj)), 'wb+'))

# Create a new Model
m = gp.Model('santa-workshop')

# Create a 5000x5 matrix with the assignments.
# asm[f][i] == 1 if family f is assigned choice i
asm = []
for f in range(NUM_FAMILY):
    asm_i = []
    for c in range(NUM_CHOICES):
        asm_i.append(m.addVar(vtype=GRB.BINARY, name=f'fam_{f}_{c}'))
    asm.append(asm_i)

# Create a 100x176x176 matrix with the occupancies.
# occ[d][npp1][npp2] == 1 if (npp1 + 125) people go to (day + 1) and 
# (npp2 + 125) people go to (day + 2)
occ = []
for d in range(NUM_DAY):
    occ_d = []
    for npp1 in range(NUM_OCCUPANCY):
        occ_d_npp1 = []
        for npp2 in range(NUM_OCCUPANCY):
            occ_d_npp1.append(m.addVar(vtype=GRB.BINARY, 
                name=f'day_{d + 1}_{npp1 + MIN_OCCUPANCY }_{npp2 + MIN_OCCUPANCY}'))
        occ_d.append(occ_d_npp1)
    occ.append(occ_d)

# Store the occupancies in a dict
occupancy = {}
for d in range(NUM_DAY):
    occupancy[d] = gp.quicksum(asm[f][c] * family_size_dict[f]
        for f in range(NUM_FAMILY) for c in range(NUM_CHOICES)
        if choices[f][c] == (d+1))

occupancy[NUM_DAY] = occupancy[NUM_DAY - 1]

# Each family should get assigned exactly one choice
for f in range(NUM_FAMILY):
    m.addConstr(gp.quicksum(asm[f][c] for c in range(NUM_CHOICES)) == 1,
        name = f'1_choice_{f}')

for d in range(NUM_DAY):
    # Occupancies should be between 125 and 300 per day
    m.addConstr(occupancy[d] >= MIN_OCCUPANCY, name = f'min_occ_{d}')
    m.addConstr(occupancy[d] <= MAX_OCUPPANCY, name = f'max_occ_{d}')
    
    # Create indicator (binary) variables corresponding to the value
    y_sum_npp1 = gp.quicksum(occ[d][npp1][npp2] * COUNT[npp1]
                             for npp1 in range(NUM_OCCUPANCY)
                                for npp2 in range(NUM_OCCUPANCY))
    y_sum_npp2 = gp.quicksum(occ[d][npp1][npp2] * COUNT[npp2]
                             for npp1 in range(NUM_OCCUPANCY) 
                                for npp2 in range(NUM_OCCUPANCY))
    m.addConstr(y_sum_npp1 == occupancy[d], name=f'indicator_npp1_{d}')
    m.addConstr(y_sum_npp2 == occupancy[d + 1], name=f'indicator_npp2_{d}')
    # Exactly 1 indicator variable should be set
    y_sum = gp.quicksum(occ[d][npp1][npp2] for npp1 in range(NUM_OCCUPANCY) 
        for npp2 in range(NUM_OCCUPANCY))
    m.addConstr(y_sum == 1, name=f'1_indicator_{d}')

# Make sure the binary indicators are consistent
for d in range(NUM_DAY - 1):
    for t in range(NUM_OCCUPANCY):
        y_sum_npp1 = gp.quicksum(occ[d][npp1][t] for npp1 in range(NUM_OCCUPANCY))
        y_sum_npp2 = gp.quicksum(occ[d + 1][t][npp2] for npp2 in range(NUM_OCCUPANCY))
        m.addConstr(y_sum_npp1 == y_sum_npp2, name='consistent_occ_{}_{}'.format(d, t))

# Initialize solution
init_occs = np.zeros(NUM_DAY + 1, dtype=int)
for f in range(NUM_FAMILY):
    init_occs[initial[f] - 1] += family_size_dict[f]
init_occs[-1] = init_occs[-2]

for f in range(NUM_FAMILY):
    for c in range(NUM_CHOICES):
        if choices[f][c] == initial[f]:
            asm[f][c].start = 1
        else:
            asm[f][c].start = 0

for d in range(NUM_DAY):
    for npp1 in range(NUM_OCCUPANCY):
        for npp2 in range(NUM_OCCUPANCY):
            if init_occs[d] == npp1 + MIN_OCCUPANCY and init_occs[d + 1] == npp2 + MIN_OCCUPANCY:
                occ[d][npp1][npp2].start = 1
            else:
                occ[d][npp1][npp2].start = 0

# Set objective
pref_cost = gp.quicksum(asm[f][c]*PREF_COSTS[f][c] for c in range(NUM_CHOICES) for f in range(NUM_FAMILY))
acc_cost = gp.quicksum(occ[d][npp1][npp2]*ACC_COSTS[npp1][npp2]
                       for d in range(NUM_DAY) for npp1 in range(NUM_OCCUPANCY) for npp2 in range(NUM_OCCUPANCY))
m.setObjective(pref_cost + acc_cost, GRB.MINIMIZE)

# Optimize model
m.optimize(print_callback)

m.setParam('MIPGap', 0)

input()

print(m.objVal)
for v in m.getVars():
    print('%s %g' % (v.varName, v.x))