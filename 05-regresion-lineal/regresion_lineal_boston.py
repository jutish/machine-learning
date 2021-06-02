# Aplicamos el analisis de regresion lineal al ejemplo planteado en 5.1
# El ejemplo 5.1 usaba el dataset load_boston el cual tiene el precio 
# de una propiedad segun sus caracteristicas. Para este ejemplo buscamos
# hacer una prediccion del precio basada en la cantidad de habitaciones promedio
# que tiene la habitacion
# Dataset de ejemplo: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
plt.style.use('ggplot')

# Cargamos los datos de entrada
boston = load_boston()
print(boston.feature_names)  # Nombre de los parametros (1x13)
print(boston.data[0:2])  # Matriz de parametros (506x13)
print(boston.target[0:2])  # Matriz de resultados o valor de la casa (506x1)
print(boston.DESCR)  # Arroja una descripcion del DataSet

# Lo paso a un pd.DataFrame solo para graficar un histograma de c/parametro
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.hist()
plt.show()

# Hago un scatter basado solo en el nro de habitaciones y los precios
# pintando de naranja aquellos valores mayores a la media de habitaciones y 
# de azul aquellos menores a la media de precios
X = np.array([df.RM])
y = np.array([boston.target])
xMean = X.mean()
color = ['blue', 'orange']
condiciones = [X > xMean][0][0] * 1
colores = [color[c] for c in condiciones]
plt.scatter(X, y, c=colores, alpha=0.7)
plt.show()

# Transponemos para no tener problemas al entrenar
X = X.T
y = y.T

# Creamos nuestro modelo
regr = linear_model.LinearRegression()

# Entrenamos el modelo
regr.fit(X, y)

# Hacemos la prediccion
y_pred = regr.predict(X)

# Veamos los coeficientes obtenidos, En nuestro caso, serán la Tangente
print('Coeficientes: ', regr.coef_)

# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: ', regr.intercept_)

# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))

# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y, y_pred))

# Ploteo datos originales vs datos predichos y la recta de regresion
# donde la recta de regresion y = a*x + b (y = regr.coef_ * x + regr.intercept_)
a = regr.coef_[0]
b = regr.intercept_
# Datos originales:
plt.title('Datos Originales')
plt.scatter(X, y, alpha=0.7)
plt.plot([4,9], [a*4 + b, a*9 + b], color='blue')
plt.show()
# Datos predichos
plt.title('Datos Predichos')
plt.scatter(X, y_pred, alpha=0.7)
plt.plot([4,9], [a*4 + b, a*9 + b], color='blue')
plt.show()

# Regresion de multiples variables
# Agregamos una variable mas para predicir el precio. En este caso CRIM que 
# mide el ratio per capita de crimenes en la zona
# Nuestra “ecuación de la Recta”, ahora pasa a ser:
# Y = b + m1 X1 + m2 X2 + … + m(n) X(n) y deja de ser una recta)
X1 = np.array([df.RM])  # Nro de habitaciones
X2 = np.array([df.CRIM])  # Ratio de criminalidad
y = np.array([boston.target])  # Valor promedio de la propiedad

# Graficamos en 3D el scatter 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X1, X2, y)
plt.show()

# Entrenamos el modelo con mas de una variable
X1X2_train = df[['RM', 'CRIM']]
z_train = y.T
regr2 = linear_model.LinearRegression()
# Entrenamos el modelo, esta vez, con 2 dimensiones
# obtendremos 2 coeficientes, para graficar un plano
regr2.fit(X1X2_train, z_train)
# Hacemos la predicción con la que tendremos puntos sobre el plano hallado
z_pred = regr2.predict(X1X2_train)
# Los coeficientes son 2 c/uno afecta a una variable predictiva (X1, X2) -> Y = b + m1 X1 + m2 X2
print(f'\nCoeficientes {regr2.coef_}')
print(f'Termino Independent {regr2.intercept_}')
# Error cuadrático medio
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
# Evaluamos el puntaje de varianza (siendo 1.0 el mejor posible)
print('Variance score: %.2f' % r2_score(z_train, z_pred))

print(X1.min(), X1.max())
print(X2.min(), X2.max())
print(z_train.min(), z_train.max())

# Ploteamos los puntos originales en 3D y el plano que intenta predecirlos
fig = plt.figure()
ax = Axes3D(fig)
# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(3, 9, num=10), np.linspace(0, 90, num=10))
# Calculamos los valores del plano para los puntos x e y
nuevoX = (regr2.coef_[0][0] * xx)
nuevoY = (regr2.coef_[0][1] * yy) 
# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + regr2.intercept_)
# Graficamos el plano predictor
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')
# Graficamos los puntos originales
ax.scatter(X1X2_train.RM, X1X2_train.CRIM, z_train, c='blue',s=30)
# Graficamos los puntos predichos en rojo
ax.scatter(X1X2_train.RM, X1X2_train.CRIM, z_pred, c='red', s=40)
# con esto situamos la "camara" con la que visualizamos
# ax.view_init(elev=30., azim=65)
ax.set_xlabel('Nro de habitaciones')
ax.set_ylabel('Ratio Criminalidad')
ax.set_zlabel('Precio Promedio')
ax.set_title('Regresión Lineal con Múltiples Variables')
plt.show()