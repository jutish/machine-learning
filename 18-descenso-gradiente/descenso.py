# https://youtu.be/-_A_AAxqzCg
# El descenso de gradiente es una tecnica de optimizacion
# buscamos minimizar el error definida por una funcion de coste no convexa.

# Importamos las librerias
import numpy as np
import scipy as sc  # Se basa en numpy y es una extension
import matplotlib.pyplot as plt

# Ejemplo de optimizar funcion. funcion_ejemplo.png
# Podemos optimizar cualquier funcion mientras sea derivable
# Nota: th es la letra griega theta y en ML se suele usar para representar
# el vector de parametros. seria como un vector que contiene la X y la Y
func = lambda th: np.sin(1 / 2 * th[0] ** 2 - 1 / 4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1])
print(func([5,3]))

# Generamos la base para poder graficar
res = 100  # Resolucion del grafico que generamos
_X = np.linspace(-2, 2, res)  # Genero un vector de 100 valores de -2 a 2
_Y = np.linspace(-2, 2, res)
_Z = np.zeros((res, res))  # Creo la matriz 100 x 100 (res x res) que tendra la salida.

# Llamamos para cada par (_X, _Y) la funcion "func" y guardamos el valor en _Z
for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        _Z[iy, ix] = func([x,y])  # _Z es una matriz. Las filas son Y y las columnas X por eso se dispone asi la asignacion.

# Impresion en 3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(_X, _Y)
surf = ax.plot_surface(X, Y, _Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

# La vista de pajaro donde se ve para el Eje X e Y y la profundidad de Z en
# forma de anillas se puede usar .contour o .contourf que muestra la forma solida.
plt.contourf(_X, _Y, _Z, 100)  # El 100 aumenta la res.
plt.colorbar()
# plt.show()


# Generamos un punto aleatorio sobre la superficie X e Y que sera donde 
# nuestro gradiente comenzara a buscar descender
theta = np.random.rand(2) * 4 - 2 # Genero dos valores entre -2 y 2
plt.plot(theta[0], theta[1], 'o', c='red')  # El punto de partida en rojo
# plt.show()

# Ahora partiendo donde estoy parado "theta" debo analizar para que lado
# comenzar a descender. Esto seria ver la pendiente para X y para Y (Long y lat.)
# y moverme en X e Y hacia dicho lado. Matematicamente se representa como 
# toma la derivada parcial de X y de Y respecto a la funcion e ir en el sentido
# contrario.

# La variable h es una pequeña cantidad en la que incremento las coordenas 
# para emular la derivada parcial. Ya que en realidad no uso la derivada parcial
# sino que uso el concepto de derivada parcial, que es ver el valor de la
# salida para un pequeño incremento en la entrada y comparar con el valor sin
# el incremento para luego si saber para donde ir. Podria haber obtenido la
# derivada de la funcion usando alguna libreria y luego si evaluarlo en X e Y
# pero en el video lo hace de esta forma y me parece bien.
h = 0.001

# Defino el vector gradiente el cual tiene 2 entradas porque nuestro punto rojo
# theta, tiene dos pendientes en latitud y longitud.
grad = np.zeros(2)

# Defino lr como learning rate que corresponde a la "importancia o peso" que
# se le da al vector gradiente (grad) en la forma de optimizar el error.
# recordemos que theta = theta - lr * grad
lr = 0.001  # 

# En theta tenemos las coordenadas del punto rojo que figura en el 
# grafico theta[0] y theta[1].
_T = np.copy(theta)

for _ in range(10000): # deberia converger pero por ahora lo hago 1000 veces
    for it, t in enumerate(theta):
        _T = np.copy(theta)
        _T[it] = _T[it] + h
        deriv = (func(_T) - func(theta)) / h  # deriv representa la pendiente
        grad[it] = deriv  # Actualizo el vector gradiente.
    theta =  theta - lr * grad  # Actualizo theta basado en el gradiente
    print(func(theta)) # Deberia ir decrementado el valor de Z
    if _ % 1000 == 0:  # cada 10 iteraciones ploteo
        plt.plot(theta[0], theta[1], 'o', c='red', alpha=0.3)

plt.plot(theta[0], theta[1], 'o', c='green')  # El punto de llegada en verde
plt.show()