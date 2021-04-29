# https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/
# Las compuertas XOR
# Tenemos dos entradas binarias (1 ó 0) y la salida será 1 sólo si una de las
# entradas es verdadera (1) y la otra falsa (0).
# Es decir que de cuatro combinaciones posibles, sólo dos tienen salida 1 y las
# otras dos serán 0, como vemos aquí:
# XOR(0,0) = 0
# XOR(0,1) = 1
# XOR(1,0) = 1
# XOR(1,1) = 0

# Utilizaremos Keras que es una librería de alto nivel, para que nos sea más
# fácil describir las capas de la red que creamos y en background es decir,
# el motor que ejecutará la red neuronal y la entrenará estará la
# implementación de Google llamada Tensorflow, que es la mejor que existe
# hoy en día.

# Utilizaremos numpy para el manejo de arrays. De Keras importamos el tipo
# de modelo Sequential y el tipo de capa Dense que es la “normal”.
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json

# Cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0], [0,1], [1,0], [1,1]])

# Y estos son los resultados en el mismo orden
target_data = np.array([[0], [1], [1], [0]], 'float32')
print('Training data: ', training_data)
print('Target data: ', target_data)

# Primero creamos un modelo vació de tipo Sequential. Este modelo se refiere
# a que crearemos una serie de capas de neuronas secuenciales,
# “una delante de otra”.
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.fit(training_data, target_data, epochs=1000)

# Evaluamos el modelo
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())

# Serializar el modelo a JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
model.save_weights("model.h5")
print("Modelo Guardado!")
 
# mas tarde...
with open('model.json','r') as f:
    json = f.read()
model = model_from_json(json)
# cargar pesos al nuevo modelo
model.load_weights("model.h5")
print("Cargado modelo desde disco.")
# Compilar modelo cargado y listo para usar.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

