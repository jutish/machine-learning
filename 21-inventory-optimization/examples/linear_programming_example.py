import pulp
from scipy.optimize import linprog
# There are four products
# x1, x2, x3 and x4
# $20*x1 + $12*x2 + $40*x3 + $25*x4 = profits per product
# x1+x2+x3+x4 can't exceed 50
# x1 = 3A
# x2 = 2A + B
# x3 = A + 2B
# x4 = 3B
# A = 100
# B = 90

# Using linprog
obj = [-1, -2]
lhs_ineq=[[2, 1],
          [-4, 5],
          [1, -2]]
rhs_ineq=[20,
          10,
          2]
lhs_eq = [[-1, 5]]  # Green constraint left side
rhs_eq = [15]       # Green constraint right side

bnd = [(0, float("inf")),  # Bounds of x
       (0, float("inf"))]  # Bounds of y

# opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
#                A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
#                method="revised simplex")

# print(opt)

# Using pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name='small-problem', sense=LpMaximize)

# Initialize the decisions variables
x = LpVariable(name='x', lowBound=0)
y = LpVariable(name='y', lowBound=0)
expression = 2 * x + 4 * y
print(type(expression))

constraint = 2 * x + 4 * y >= 8
print(type(constraint))

# Add the constraints to the model
model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

# Add the objective function to the model
model += x + 2 * y
# print(model)

status = model.solve()


print(f"status: {model.status}, {LpStatus[model.status]}")

print(f"objective: {model.objective.value()}")

for var in model.variables():
     print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
     print(f"{name}: {constraint.value()}")

# model.objective holds the value of the objective function, model.constraints
# contains the values of the slack variables, and the objects x and y have the
# optimal values of the decision variables. model.variables() returns a list
# with the decision variables:

print(model.variables())