import pandas as pd
from datetime import date 
# Read Excel and split into 2 data frames. Stock and Orders
sheets = {'stock':'Stock 11-06', 'orders':'Ordenes de Venta 11-06'}

orders = pd.read_excel('Datos para pasar a IA.xlsx', sheet_name=sheets['orders'])
stock = pd.read_excel('Datos para pasar a IA.xlsx', sheet_name=sheets['stock'])

# Rename columns for an easier use
orders.columns = ['order_type','order_id','article_id','article','client_id',
                 'ordered_qty']
stock.columns = ['deposit_id','article_id','article','lot',
                'exp_date','box_qty']

# Join orders articles and stock qty
orders_stock = pd.merge(orders, stock, on=['article_id'])
orders_stock['covered_perct'] = orders_stock['box_qty'] / orders_stock['ordered_qty']
orders_stock['exp_days'] = orders_stock['exp_date'] - pd.Timestamp.today()
print(orders_stock[['article_id','ordered_qty','box_qty', 'covered_perct']])

orders_stock.to_csv('orders_stock.csv')
