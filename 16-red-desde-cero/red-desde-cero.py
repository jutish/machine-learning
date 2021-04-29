import numpy as np


# Es la funcion de activacion
# Caso especial de función logística y definida por: g(z) = 1/(1+e-z)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Su derivada puede ser expresada como g'(t) = g(t)(1 – g(t)).
def sigmoid_derivada(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Tangente hiperbolica
def tanh(x):
    return np.tanh(x)


# Derivada de la tangente hiperbolica --> arco-tangente = 1 - tanh^2 x
def tanh_derivada(x):
    return 1.0 - tanh(x)**2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        # Inicializo los pesos
        self.weights = []
        self.deltas = []

        # Asigno los pesos para las capas de entrada y la capa oculta
        # recordemos que ambas capas tienen una neurona extra de bias
        # que no se diagrama pero se tiene en cuenta.
        # La funcion np.random.random((tupla)) ej: random((2,3)) genera
        # una matriz de 2 filas x 3 columnas con valores random entre 0 y 1
        # de esta formas obtengo por cada neurona el peso con las neuronas 
        # de la siguiente capa.
        # Donde layers = [2,3,2] -> no se muestra pero se considera 
        # las neuronas de bias
        for i in range(1, len(layers) - 1):
            # Asigno los pesos entre la capa de entrada y la capa oculta
            # Multiplico por 2 y resto 1 asi valores quedan entre -1 y 1
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
            # Asigno los pesos entre la capa oculta y la de salida
            r = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
            self.weights.append(r)


    def fit(self,X, y, learning_rate=0.2, epochs=100000):
        # Agrego columna de unos a las entradas X
        # Con esto agregamos la unidad de Bias (neurona extra) 
        # a la capa de entrada
        # Ej si la X es [[0,0],[0,1],...,[-1,1]]
        # el res. es X = [[1,0,0],[1,0,1],...,[1,-1,1]]
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # Int entre 0 y nro filas de X
            a = [X[i]]  # Elijo una entrada al azar.dada por i
            for l in range(len(self.weights)):
                # dot_value = np.dot(a[0],self.weights[l])  # 1x3 * 3x4 = 1x4
                print(self.weights[0])

nn = NeuralNetwork([2,3,2],activation ='tanh')
X = np.array([[0, 0],   # sin obstaculos
              [0, 1],   # sin obstaculos
              [0, -1],  # sin obstaculos
              [0.5, 1], # obstaculo detectado a derecha
              [0.5,-1], # obstaculo a izq
              [1,1],    # demasiado cerca a derecha
              [1,-1]])  # demasiado cerca a izq
 
y = np.array([[0,1],    # avanzar
              [0,1],    # avanzar
              [0,1],    # avanzar
              [-1,1],   # giro izquierda
              [1,1],    # giro derecha
              [0,-1],   # retroceder
              [0,-1]])  # retroceder
nn.fit(X, y, learning_rate=0.03,epochs=1)