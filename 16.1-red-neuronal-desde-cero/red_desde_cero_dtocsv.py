import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import warnings
warnings.filterwarnings("error")

# Creamos el dataset
n = 500  # Nro de registros/observaciones tenemos en nuestro dataset
p = 2  # Cuantas caracteristicas/parametros tiene cada registro
# X Tendrá 500 puntos de entrada de la forma [X1, X2] que ploteados
# forman 2 circulos o nubes circulares de puntos
# Y Dice a que clase/circulo (1, 0) pertenece cada punto
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = np.array([Y]).T # Llevo la dimension de Y a (500,1) antes era (500,)
# plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c='skyblue')
# plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c='salmon')
# plt.axis('equal')
# plt.show()


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
# relu = lambda x: np.max(0, x)  # Funcion RELU
_x = np.linspace(-5, 5, 100)
# plt.plot(_x, sigm[0](_x))  # Funcion sigmoide
# plt.show()
# plt.plot(_x, sigm[1](_x))  # Derivada de la funcion sigmoide
# plt.show()


# Genero mi arquitectura
# Creo una funcion que lo hace por mi.
# Donde topology va a ser algo del tipo -> topology = [2, 4, 8, 16, 8, 4, 1]
def createNN(topology, actFunction):
    nn = []  # Contiene cada una de las capas ocultas que forman la red.
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(topology[l], topology[l + 1], actFunction))
    return nn


# Defino la funcion de coste
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
# train: Si esta variable esta activa el algoritmo aplica el Backward pass
#        y el descenso del gradiente. Es decir que se entrena el modelo
#        al estar desactivado no se aplica lo anterior y solo se estaria 
#        usando el metodo para predecir.

def train(neuralNet, X, Y, l2Cost, lr=0.5, train=True):
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
    # out = [(z0, a0), (z1, a1)]
    out = [(None, X)]  # La activacion de la capa 1 son los datos de entrada X
    for l, layer in enumerate(neuralNet):
        # out[-1][1] toma el ultimo output y se queda con la activacion la cual es la entrada de la sig. capa
        z = out[-1][1] @ neuralNet[l].W + neuralNet[l].b
        a = neuralNet[l].actFunction[0](z)
        out.append((z, a))
    # print(l2Cost[0](out[-1][1], Y))  # out[-1][1] tiene el valor de la funcion de activ. de la ultima capa que es igual a Yp. (salida propuesta)
    if train:
        # Backward pass (backpropagation)
        deltas = []
        for l in reversed(range(0,len(neuralNet))):
            z = out[l+1][0]
            a = out[l+1][1]
            # Si estamos en la ultima capa
            if l == len(neuralNet) - 1:
                # Calcular delta ultima capa
                termino11 = l2Cost[1](a, Y)  # Derivada parcial de la funcion de activacion respecto al coste = aplicar la derivada del coste en el resultado de la activacion
                termino21 = neuralNet[l].actFunction[1](a)  # Derivada parcial de la suma ponderada "z" respecto a la funcion de activacion = aplicar la derivada de la funcion de activacion en el resultado de "z"
                # print(f'capa{l}: ',termino11.shape, termino21.shape)
                # delta = termino11 * termino21
                delta = l2Cost[1](a, Y) * neuralNet[l].actFunction[1](a)  # en el video pone a en lugar de z
                deltas.insert(0, delta)
            else:
                # Calcular delta capa previa
                termino12 = _W.T
                termino22 = deltas[0]
                termino32 = neuralNet[l].actFunction[1](a)
                # delta = termino12 @ (termino22 * termino32).T
                delta =  deltas[0] @ _W.T * neuralNet[l].actFunction[1](a)  # en el video pone a en lugar de z
                # print(delta.shape)
                deltas.insert(0, delta)
                
            _W = neuralNet[l].W  # Salvo el valor de los parametros antes de actualizarlos en el descenso del gradiente.
            # Gradient descent
            termino43 = neuralNet[l].b
            termino44 = np.mean(deltas[0], axis=0, keepdims=True)
            # print(f'Gradient descendt capa b:{l}:', termino43.shape, termino44.shape)
            # neuralNet[l].b = termino43 - termino44 * lr
            neuralNet[l].b = neuralNet[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            termino13 = neuralNet[l].W
            termino23 = out[l][1]
            termino33 = deltas[0]
            # print(f'Gradient descendt capa{l}:', termino13.shape, termino23.shape, termino33.shape)
            # neuralNet[l].W = termino13 - termino23 @ termino33 * lr
            neuralNet[l].W = neuralNet[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]  # Retorno la prediccion

#  Hasta aca la red. Ahora lo siguiente es usarla y plotear

import time
from IPython.display import clear_output
# Creo la red neuronal
iteraciones = 1000
topology = [p, 4, 8 ,1]  # Defino el numero de neuronas por capa
neuralNet = createNN(topology, sigm)
loss = []

res = 50
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)
_Y = np.zeros((res, res))
for i in range(iteraciones):
    Yp = train(neuralNet, X, Y, l2Cost, lr=0.05, train=True)
    if i % 25 == 0:
        loss.append(l2Cost[0](Yp, Y))
        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neuralNet, np.array([[x0, x1]]), Y, l2Cost, train=False)[0][0]
        
        # print(len(loss), loss)
        plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c='skyblue')
        plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c='salmon')
        plt.pcolormesh(_x0, _x1, _Y, cmap='coolwarm', shading='auto')
        plt.axis('equal')
        plt.show()
        time.sleep(0.5)
        plt.close()

# plt.plot(range(len(loss)), loss)
# plt.show()
# print(loss)
