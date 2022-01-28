import gurobipy as grb
from gurobipy import *

R = ['Carlos', 'Joe', 'Monika']
J = ['Tester', 'JavaDeveloper', 'Architect']

combinations, ms = multidict({
    ('Carlos','Tester'):53,
    ('Carlos','JavaDeveloper'):27,
    ('Carlos','Architect'):13,
    ('Joe','Tester'):80,
    ('Joe','JavaDeveloper'):47,
    ('Joe','Architect'):67,
    ('Monika','Tester'):53,
    ('Monika','JavaDeveloper'):73,
    ('Monika','Architect'):47
    })

# Define model
m = Model('RAP')

# Define variables
x = m.addVars(combinations, name='assign')

# Define constraints
jobs = m.addConstrs((x.sum('*',j) == 1 for j in J), name='job')
resources = m.addConstrs((x.sum(r,'*') <= 1 for r in R), name='resource')

# Set objective
m.setObjective(x.prod(ms), GRB.MAXIMIZE)

# Write model formulation
m.write('RAP.lp')

# Run optimization engine
m.optimize()

# Display optimal values of decision variables
for v in m.getVars():
    if(abs(v.x) > 1e-6):
        print(v.varName, v.x)

# Display optimal total machine score
print('Total Machine Score: ', m.objVal)
