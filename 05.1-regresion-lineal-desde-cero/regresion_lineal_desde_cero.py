# Ejemplo basado en el canal de youtube dotcsv
# https://www.youtube.com/watch?v=w2RJ1D6kz-o
# Expresando el error cuadratico medio "media((Yr-Ye)^2)"
# en forma de vectores (Y-XW).T (Y-XW), igualando
# a cero y derivando, minimizamos el error y obtenemos la funcion en forma
# vectorial [W = (X.T X)^-1 X.T Y]

# Nota: Yr es el valor real, Ye es el valor esperado
# Nota2: W es la matriz de parametros o bien la pendiente de la recta
#        tambien incluye el termino independiente
#        X es la matriz de entrada cada columna representa una caracteristica
#        y cada fila una observacion
#        Y es la matriz de salida
#        Al resolver W = (X.T X)^-1 X.T Y obtenemos los pesos/pendientes
#        de las rectas, planos o hiperplanos que minimizan el error para un
#        conjunto de puntos/entradas dadas X. Obtemos los parametros.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Cargamos la libreria
boston = load_boston()
print(boston.DESCR)

# Cargamos nuestra matriz de entrada
X = boston.data[:,5]  # RM esta en la posicion 5. Numero promedio de habit.
Y = boston.target
# Ploteamos
plt.scatter(X, Y, alpha=0.3)  #Alpha es la transp. permite ver concentracion
# Agregamos a X una columna de 1(unos) para representar el termino indep.
X = np.array([np.ones(len(X)),X]).T
print(X.shape)
# Obtenemos las pendientes o valor de la matriz W segun la formula 
# W = (X.T X)^-1 X.T, tambien se suele llamar Beta al error (B)
W = np.linalg.inv(X.T @ X) @ X.T @ Y
print(f'y = w0 + w1*x1 => y = {W[0]} + {W[1]}*x1')
# Graficamos la linea basados en y = w0 + w1*x1
# Usamos dos puntos de X en este caso 4 y 9 luego la salida Y queda dada
# por la formula de la recta
plt.plot([4,9], [W[0]+W[1]*4, W[0]+W[1]*9], color='red')
plt.show()

#Se me ocurrio obtener el error cuadratico medio  (Y-XW).T (Y-XW)
error = (Y-X@W).T @ (Y-X@W)
print('Error cuadratico medio: ',error)


