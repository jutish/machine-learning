"""
The simplified Whiskas Model Python Formulation for PuLP Modeller
Author: Esteban Marcelloni, 2021
Source: https://coin-or.github.io/pulp/CaseStudies/a_blending_problem.html
"""

# Import PuLP modeler functions
from pulp import *

# Create the 'prob' variable to contain the problem data
prob = LpProblem('The_Whiskas_Problem', LpMinimize)

# The 2 variables Beef and Chicken are created with a lower limit of zero
x1 = LpVariable('Chicken_Percent', 0, None, LpInteger)
x2 = LpVariable('Beef_Percent',0)

# The objective function is added to 'prob' first
prob += (0.013 * x1 + 0.008 * x2, 'Total cost of ingredients per can')

# The five constraints are entered
prob += (x1 + x2 == 100, 'Percentages_Sum')
prob += (0.100 * x1 + 0.200 * x2 >= 8.0, 'Protein_Req.')
prob += (0.080 * x1 + 0.100 * x2 >= 6.0, 'Fat_Req.')
prob += (0.001 * x1 + 0.005 * x2 <= 2.0, 'Fibre_Req.')
prob += (0.002 * x1 + 0.005 * x2 <= 0.4, 'Salt_Req.')

# Check our model
print(prob)

# Save the model to a .lp file
prob.writeLP('Whiskas_Model.lp')

# Solve
prob.solve()

# Firstly, we request the status of the solution, which can be one of 
# “Not Solved”, “Infeasible”, “Unbounded”, “Undefined” or “Optimal”.
# The status of the solution is printed to the screen
print('Status:', LpStatus[prob.status])

# The variables and their resolved optimum values can now be printed to the screen.
# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, ' = ', v.varValue)

# The objective is printed 
print('Total Cost of ingredients per can = ',value(prob.objective))

# Full formulation of the problem


