import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Creamos el dataset
n = 500  # Nro de registros/observaciones tenemos en nuestro dataset
p = 2  # Cuantas caracteristicas/parametros tiene cada registro
# X Tiene 500 puntos de entrada de la forma [X1, X2] que ploteados
# forman 2 circulos o nubes circulares de puntos
# Y Dice a que clase/circulo (1, 0) pertenece cada punto
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='skyblue')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='salmon')
plt.axis('equal')
plt.show()


# Clase de la capa de la red
# b es un arreglo de (1 x nNeur) inicializado con valores rand. entre -1 y 1
#   este arreglo representa el valor del Bias por cada neurona.
# W es un arreglo de (nConnection x nNeur) inicializado con valores rand entre -1 y 1
#   este arreglo representa el peso de cada conexion por neurona de la capa
class NeuralLayer():
    def __init__(self, nConnection, nNeur, actFunction):
        self.actFunction = actFunction
        self.b = np.random.rand(1, nNeur) * 2 - 1  # Los valores quedan entre -1 y 1
        self.W = np.random.rand(nConnection, nNeur) * 2 - 1  # Idem


# Funcion de activacion
sigm = (lambda x: 1 / (1 + np.e ** (-x)),  # sigm[0] => Funcion Sigmoide
        lambda x: x * (1 - x))  # sigm[1] => Derivada funcion Sigmoide
relu = lambda x: np.max(0, x)  # Funcion RELU
_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x))  # Funcion sigmoide
plt.show()
plt.plot(_x, sigm[1](_x))  # Derivada de la funcion sigmoide
plt.show()


# Genero mi arquitectura
# Creo una funcion que lo hace por mi.
def createNN(topology, actFunction):
    nn = []  # Contiene cada una de las capas ocultas que forman la red.
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(topology[l], topology[l + 1], actFunction))
    return nn

# Creo la red neuronal
topology = [p, 4, 8, 16, 8, 4, 1]  # Defino el numero de neuronas por capa
neuralNet = createNN(topology, sigm)

# l2Cost[0]: Es la funcion de coste del valor cuadratico medio
# l2Cost[1]: Es la derivada de la funcion de coste
# Yp: Valor predicho
# Yr: Valor real
l2Cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
          lambda Yp, Yr: (Yp - Yr))

# En el metodo train sucede todo.
# neuralNet: Es la red neuronal creada
# X: Los parametros de entrada en este caso el par [X1, X2] que representa un
#    punto en un plano cartesiano
# Y: Es la salida esperada. En este caso puede ser 1 o 0 segun si los puntos
#    de la entrada corresponden a una nube de puntos o a otra
# lr: Es el Learning Rate el cual afecta al gradiente.
def train(neuralNet, X, Y, l2Cost, lr=0.5):

    # Forward pass
    # z es el resultado de aplicar la funcion lineal/suma ponderada
    # a cada una de las 500 entradas del ejemplo capa por capa
    # donde X(500x2) .W(2x4) .b(1x4) => X@W = 500x4 + b => z (500x4)
    # z es el resultado de la suma pondedara de cada entrada en cada una de las
    # neuronas. z = X @ neuralNet[0].W + neuralNet[0].b 
    # Luego lo paso por la funcion de activacion, obtengo a (500x4) pero donde
    # la suma ponderada para cada entrada x capa se le aplico la sigmoide en este caso.
    # a = neuralNet[0].actFunction[0](z)
    # Para cada capa la salida de una es la entrada de la otra, excepto la primera
    # cuya entrada es X. En out[()] guarda el valor de z y a
    # out = [(z0, a0), (z1, a)]
    out = [(None, X)]  # La activacion de la capa 1 son los datos de entrada X
    for l, layer in enumerate(neuralNet):
        # out[-1][1] toma el ultimo output y se queda con la activacion la cual es la entrada de la sig. capa
        z = out[-1][1] @ neuralNet[l].W + neuralNet[l].b
        a = neuralNet[l].actFunction[0](z)
        out.append((z, a))
    print(out[-1][1])

train(neuralNet, X, Y, l2Cost, 0.5)
