"""
AGD Inventory Optimization
Author: Esteban Marcelloni, 10th Oct. 2021
"""

import psutil as ps
import pandas as pd
import numpy as np
from datetime import date
from pulp import *

p = ps.Process()

# Read Excel and split into 2 data frames. Orders and Stock
sheets = {'stock':'Stock 11-06', 'orders':'Ordenes de Venta 11-06'}
# path = 'Datos para pasar a IA - 2 ordenes.xlsx'
path = 'Datos para pasar a IA.xlsx'
orders = pd.read_excel(path, sheet_name=sheets['orders'])
stock = pd.read_excel(path, sheet_name=sheets['stock'])

# Rename columns for an easier use
orders.columns = ['order_type','order_id','article_id','article','client_id',
                 'ordered_qty']
stock.columns = ['deposit_id','article_id','article','lot',
                'exp_date','box_qty']

# Calculate total qty per order
orders['total_order_qty'] = orders.groupby('order_id').ordered_qty.transform('sum')

# Calculate total available stock per product
stock['total_product_stock'] = stock.groupby('article_id').box_qty.transform('sum')

# Add a unique key to reference lot number after combine with orders.
stock['lot_unique'] = stock.index

# Set correct types
orders = orders.astype({'order_type':'string','order_id':'string',
            'article_id':'string', 'article':'string', 'client_id':'string',
            'ordered_qty':'int64', 'total_order_qty':'int64'})

stock = stock.astype({'deposit_id':'string', 'article_id':'string',
    'article':'string', 'lot':'string', 'exp_date':'datetime64[ns]',
    'box_qty':'int64', 'total_product_stock':'int64','lot_unique':'int64'})

# Join orders articles and stock qty
orders_stock = pd.merge(orders, stock, on=['article_id'])

# Calculate Expiration dates betwen a lot and the actual date
orders_stock['exp_days'] = orders_stock['exp_date'] - pd.Timestamp.today()
orders_stock['exp_days'] = orders_stock.exp_days.astype('timedelta64[D]').astype('int')

# Add a column with the names of the LpVariables for an easier use later
orders_stock['lp_var'] = orders_stock.order_id.astype('str') + '_'+ \
                         orders_stock.article_id.astype('str') + '_' + \
                         orders_stock.lot_unique.astype('str')

# Remove, out of date products.
orders_stock = orders_stock[orders_stock['exp_days'] > 0]

# Write orders_stock into a *.csv
orders_stock.to_csv('orders_stock.csv', sep=';')

#***********************************************************
#************** Model Definition using PuLP ****************
#***********************************************************

solver_list = listSolvers(onlyAvailable=True)
print(solver_list)


# # Create LpVariables
# # Each variable is defined as order_id-article_id-lot_unique
# # The values of variables are between 0 and Infinite and they are Integer
variables = orders_stock['lp_var'].to_numpy()
lp_variables = LpVariable.dicts('OrderArticleLot', variables, 0, None, LpInteger)

# # Define the Objective Function

# Objective 1: Maximize the use of articles with less expiration days
# exp_days = dict(zip(orders_stock['lp_var'], -1 * orders_stock['exp_days']))
# model = LpProblem("AGD_Inventory_Optimization", LpMaximize)
# model += (lpSum([exp_days[i] * lp_variables[i] for i in variables]), 
#         'Total Remain Expiration Days')

# # Objective 2: Maximize the number of articles delivered
model = LpProblem("AGD_Inventory_Optimization", LpMaximize, )
model += (lpSum([lp_variables[i] for i in variables]), 
    'Maximize total number of articles delivered')

# Define restrictions

# Restriction 1
# Each lot has a limit (box_qty) of products. We can't supply more than we have.
restrictions = dict(zip(orders_stock['lp_var'], orders_stock['box_qty']))
for key, value in restrictions.items():
    model += (lp_variables[key] <= value, f'Lot_Articles_Limit: {key}')

# Restriction 2
# Each lot has a limit (box_qty). The total articles delivered for one lot into
# all the orders can't be more than the lot limit (box_qty)
grouped = orders_stock.groupby(['lot_unique', 'article_id'])
for name, group in grouped:
    group_vars = group['lp_var'].to_numpy()
    group_total = group['box_qty'].max()
    model += (lpSum([lp_variables[i] for i in group_vars]) <= 
        group_total , f'Lot_All_Articles_Limit_Order_ N° {name}')

# Restriction 3
# Each product has a max limit in the qty ordered (ordered_qty) per order.
# We can't assign more articles than the ordered.
grouped = orders_stock.groupby(['order_id', 'article_id'])
for name, group in grouped:
    group_vars = group['lp_var'].to_numpy()
    group_total = group['ordered_qty'].max()
    model += (lpSum([lp_variables[i] for i in group_vars]) <= 
        group_total , f'Article Limit Order N° {name}')

# Restriction 4
# Each order must be completed at least in 70%. The total articles ordered per
# order defined in "total_order_qty"
grouped = orders_stock.groupby('order_id')
for name, group in grouped:
    min_perc = 1 # At least 70%
    group_vars = group['lp_var'].to_numpy()
    group_total = group['total_order_qty'].max() 
    model += (lpSum([lp_variables[i] for i in group_vars]) >= min_perc * 
        group_total , f'Percentage Limit Order N° {name}')

# Restriction 5
# The total ordered from an Article in all the orders must be less or equal to 
# the total stock available for that article. "total_product_stock"
grouped = orders_stock.groupby('article_id')
for name, group in grouped:
    group_vars = group['lp_var'].to_numpy()
    group_total = group['total_product_stock'].max() 
    model += (lpSum([lp_variables[i] for i in group_vars]) <= group_total , 
        f'Stock Limit Order N° {name}')

# Save the model to analyze in a visual way
model.writeLP("AGD_Inventory_Optimization.lp")

# The problem is solved using PuLP's choice of Solver
# solver = getSolver('GUROBI')
model.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])

# Get a DataFrame with the propose solution
solution_df = pd.DataFrame(columns=['order_id','article_id','lot_id', 'qty'])
for v in model.variables():
    if v.varValue!=0:
        data = v.name.split('_')
        order = data[1]
        article = data[2]
        lot = data[3]
        qty = v.varValue
        row = {'order_id':order, 'article_id':article, 'lot_id':lot, 'qty': qty}
        solution_df = solution_df.append(row, ignore_index=True)
        # print(v.name,' ',v.varValue)

# Set correct types to the solution
solution_df = solution_df.astype({'order_id':'string', 'article_id':'string',
                                  'lot_id':'int64'})

# Merge orders with the propose solution. We use a LEFT JOIN to see all the 
# articles in an order even though this article doesn't have a solution.
sol = pd.merge(orders, solution_df, left_on=['order_id','article_id'], 
                                    right_on=['order_id','article_id'],
                                    how='left')
sol = pd.merge(sol, stock, left_on=['lot_id'], right_on=['lot_unique'], 
    how='left')

# Keep only the Date part of DateTime
sol['exp_date'] = sol['exp_date'].dt.date

# Calculate how many articles remains in each Lot
sol['art_cumsum'] = sol.groupby(['lot_id','article_id_x']).qty.cumsum()
sol['remain_qty'] = sol['box_qty'] - sol['art_cumsum']

# Caculate de difference between the qty ordered from an article and the qty
# supply for a Lot.
# sol['diff'] = sol['ordered_qty'] - sol['qty']

# Drop unuseful columns
sol.drop(columns=['order_type','total_order_qty','article_id_y','article_y',
    'lot_unique', 'total_product_stock','art_cumsum'], inplace=True)

# Set index to write excel
sol.set_index(['order_id','article_id_x'], drop=True, inplace=True)

# Write Excel
sol.to_excel('AGD_Inventory_Optimization.xlsx')

info = p.as_dict(attrs=['pid','name','username','memory_info','memory_percent','cpu_times'])
print(info)
