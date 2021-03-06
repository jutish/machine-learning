import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Creamos el dataset
n = 500  # Nro de registros/observaciones tenemos en nuestro dataset
p = 2  # Cuantas caracteristicas/parametros tiene cada registro
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = np.array([Y]).T # Llevo la dimension de Y a (500,1) antes era (500,)

#Imprimo el dataset del problema en 2d
plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c='skyblue')
plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c='salmon')
plt.axis('equal')
plt.show()

# Imprimo el dataset del problema en 3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1],Y[Y[:,0] == 0], c='skyblue')
ax.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1],Y[Y[:,0] == 1], c='salmon')
# ax.scatter(X[:,0],X[:,1],Y[:,0])
plt.show()


class NeuralLayer():
    def __init__(self, nConnection, nNeur, actFunction):
        self.actFunction = actFunction
        self.b = np.random.rand(1, nNeur) * 2 - 1  # Los valores quedan entre -1 y 1
        self.W = np.random.rand(nConnection, nNeur) * 2 - 1  # Idem


# Funcion de activacion
sigm = (lambda x: 1 / (1 + np.e ** (-x)),  # sigm[0] => Funcion Sigmoide
        lambda x: x * (1 - x))  # sigm[1] => Derivada funcion Sigmoide

# Defino la funcion de coste
l2Cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
          lambda Yp, Yr: (Yp - Yr))


def createNN(topology, actFunction):
    nn = []  # Contiene cada una de las capas ocultas que forman la red.
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(topology[l], topology[l + 1], actFunction))
    return nn


def train(neuralNet, X, Y, l2Cost, lr=0.5, train=True):
    # Forward pass
    out = [(None, X)]
    for l, layer in enumerate(neuralNet):
        z = out[-1][1] @ neuralNet[l].W + neuralNet[l].b
        a = neuralNet[l].actFunction[0](z)
        out.append((z, a))
    if train:
        # Backward pass (backpropagation)
        deltas = []
        for l in reversed(range(0,len(neuralNet))):
            z = out[l+1][0]
            a = out[l+1][1]
            # Si estamos en la ultima capa
            if l == len(neuralNet) - 1:
                # Calcular delta ultima capa
                delta = l2Cost[1](a, Y) * neuralNet[l].actFunction[1](a)
                deltas.insert(0, delta)
            else:
                # Calcular delta capa previa
                delta =  deltas[0] @ _W.T * neuralNet[l].actFunction[1](a)
                deltas.insert(0, delta)
            _W = neuralNet[l].W
            # Gradient descent
            neuralNet[l].b = neuralNet[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neuralNet[l].W = neuralNet[l].W - out[l][1].T @ deltas[0] * lr
    return out[-1][1]  # Retorno la prediccion


#  Hasta aca la red neuronal ahora la llamo y ploteo

# Creo la red neuronal
iteraciones = 2000
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
        

plt.pcolormesh(_x0, _x1, _Y, cmap='coolwarm', shading='auto')
plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c='skyblue')
plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c='salmon')
plt.axis('equal')
plt.show()


# Impresion 3d de la solucion
from matplotlib import cm
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x0, x1 = np.meshgrid(_x0, _x1)
surf = ax.plot_surface(x0, x1, _Y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

# Imprimo la evolucion del aprendizaje basado en el error devuelto por las predicciones
plt.plot(range(len(loss)), loss)
plt.show()
# print(loss)
