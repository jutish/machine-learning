import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Importo la data
path = './recursos/train.csv'
data = pd.read_csv(path)

# Describo
print(data.describe())

# Separo X (datos de entrada) e y (salida/precio)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Creo el modelo de arbol de desicion sin especificar la altura maxima
tree_model = DecisionTreeRegressor(random_state=1)
# Entreno el arbol
tree_model.fit(train_X, train_y)
# Hago una prediccion
tree_pred = tree_model.predict(val_X)
# Veo el error MAE (Mean Absolute Error)
mae = mean_absolute_error(tree_pred, val_y)
print('Decision Tree Regressor sin optimizar - Mean Absolute Error:', mae)

# Optimizo el parametro max_depth en busqueda de la mejor profundidad del arbol
depths = {}
for depth in range(2,50):
    tree_model = DecisionTreeRegressor(random_state=1, max_depth=depth)
    tree_model.fit(train_X, train_y)
    tree_pred = tree_model.predict(val_X)
    mae = mean_absolute_error(tree_pred, val_y)
    depths[depth] = mae
best_depth = min(depths, key=depths.get)
print(f'Optimizando max_depth - Minimo MAE: {depths[best_depth]} - Key: {best_depth}')
plt.figure().suptitle('Optimización max_depth')
plt.plot(depths.keys(), depths.values())
plt.vlines(best_depth, ymin=0, ymax=max(depths.values()), colors='red')
plt.show()

# Optimizo el parametro max_leaf_nodes en busqueda del mejor nro de hojas
leaves = {}
for leaf in range(2,200):
    tree_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=leaf)
    tree_model.fit(train_X, train_y)
    tree_pred = tree_model.predict(val_X)
    mae = mean_absolute_error(tree_pred, val_y)
    leaves[leaf] = mae
best_leaves_number = min(leaves, key=leaves.get)
print(f'Optimizando max_leaf_nodes - Minimo MAE: {leaves[best_leaves_number]} - Key: {best_leaves_number}')
plt.figure().suptitle('Optimizacion max_leaf_nodes')
plt.plot(leaves.keys(), leaves.values())
plt.vlines(best_leaves_number, ymin=0, ymax=max(leaves.values()), colors='red')
plt.show()
# Creo el arbol con los max_depth y max_leaf_nodes optimizado por separado
tree_model = DecisionTreeRegressor(random_state=1, max_depth=best_depth,
                                   max_leaf_nodes=best_leaves_number)
tree_model.fit(train_X, train_y)
tree_pred = tree_model.predict(val_X)
mae = mean_absolute_error(tree_pred, val_y)
print(f'Optimizando max_depth {best_depth} y max_leaf_nodes por separado {best_leaves_number} Minimo MAE: {mae}')

# Optimizo ambos parametros a la vez
_X = np.linspace(2, 102, 100, dtype=int)
_Y = np.linspace(2, 102, 100, dtype=int)
_Z = np.zeros((_X.shape[0],_Y.shape[0]))
best_mae = 0
best_depth = 0
best_leaf = 0
for ix, depth in enumerate(_X):
    for iy, leaf in enumerate(_Y):
        tree_model = DecisionTreeRegressor(random_state=1,
                                           max_depth=depth,
                                           max_leaf_nodes=leaf)
        tree_model.fit(train_X, train_y)
        tree_pred = tree_model.predict(val_X)
        mae = mean_absolute_error(tree_pred, val_y)
        _Z[ix][iy] = mae
        if (ix==1 and iy==1) or (mae < best_mae):
            best_mae = mae
            best_depth = depth
            best_leaf = leaf
print(f'Optimizando max_depth {best_depth} y max_leaf_nodes en conjunto {best_leaf} Minimo MAE: {best_mae}')

# Ploteo el MAE en base de los parametros max_depth y max_leaf_nodes
from matplotlib import cm
from matplotlib.ticker import LinearLocator
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(_X, _Y)
surf = ax.plot_surface(X, Y, _Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

# Uso un modelo de random forest
random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(train_X, train_y)
random_pred = random_forest_model.predict(val_X)
mae = mean_absolute_error(random_pred, val_y)
print('Random Forest Tree sin optimizar - MAE:', mae)
