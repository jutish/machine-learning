# Trucar inception V3 que es un clasificador de imagenes de google
# fue entrenada sobre imagenet. Clasifica en mas de 1000 clases

# Cargamos la librerias
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.inception_v3 import InceptionV3  #Cargamos el modelo
from keras.applications.inception_v3 import decode_predictions
from keras import backend as K
from keras.preprocessing import image
from PIL import Image

# Para poder usar el metodo K.gradients() que ya es viejo y deprecated
# Debo poner la siguiente linea de codigo.
# De usar tensor flow 2 
tf.compat.v1.disable_eager_execution()

# Cargamos el modelo dentro de una variable
iv3 = InceptionV3()
print(iv3.summary())

# Cargamos una imagen para ver si la clasifica correctamente
# En lugar de usarla como fichero la cargamos como matriz de valores.
x = image.img_to_array(image.load_img('beer.jpg'))
print('Imagen original:', x.shape)  #Alto 256px ancho 256px y prof 3 (rgb)

# El modelo InceptionV3 necesita que la imagen sea de 299px * 299px
x = image.img_to_array(image.load_img('beer.jpg', target_size=(299, 299)))
print('Imagen de 299x299:', x.shape)  #Ahora las dimensiones son 299x299 Px
print('Pixel (0,0)', x[0][0][:])  # Veo para el pixel 0,0 su intesidad en RGB

# La matriz representa la intesindad de RGB por cada pixel. Esta va de 0 a 255
# pero el modelo InceptionV3 usa un formato donde la intensidad va de -1 a 1
x /= 255  # x queda entre 0 y 1
x -= 0.5  # x queda entre -0.5 y 0.5. Centrando asi los valores
x *= 2    # x queda entre -1 y 1
print('Intensidad entre -1 y 1:', x)

# Ademas el modelo espera una dimension mas. Donde podemos pasar mas de una 
# imagen al mismo tiempo. Cada imagen constara de sus 3 dimensiones.
# en este caso tenemos solo una imagen, a esta entrada a la red se la llama
# tensor
x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
print(x.shape)

# Hago una prediccion basado en x.y obtengo un vector (1,1000)
# lo que obtengo es la capa de salida y por cada imagen, en este caso 1,
# obtengo la probabilidad que dicha imagen corresponda a 1 de las 1000 
# clasificaciones posibles que maneja el modelo.
y = iv3.predict(x)
print('Capa de salida', y.shape, y[0][0:5])

# uso decode_predictions(y) para interpretar la salida y vemos que funciona
print('Prediccion Salida: ', decode_predictions(y))

# Ataques adversarios: La idea es manipular la entrada (la imagen) buscando
# que la red/modelo la identifique como otra clase que nosotros deseamos.

# Sacamos la capa de entrada y salida
inp_layer = iv3.layers[0].input  # Capa 0 tomo la entrada.
out_layer = iv3.layers[-1].output  # Ultima capa tomo la salida.
print('imp_layer: ', inp_layer)
print('out_layer: ', out_layer)

# Difinimos la funcion de coste loss
target_class = 951  # Es un limon
loss = out_layer[0, target_class]  # Del vector de prob. queremos maximizar 951

# Calculamos el gradiente que nos va a indicar como tocar los pixeles para
# obtener el menor error. En este caso usamos el gradiente sobre la imagen de
# entrada.dada por inp_layer
# tensorflow.compat.v1.disable_eager_execution()
grad = K.gradients(loss, inp_layer)[0]
# grad = K.GradientTape(loss, inp_layer)[0]

# Definimos la funcion optimize_gradiente del tipo funcion keras.
# La entrada es inp_layer y K.learning_phase() la cual indica a Keras que 
# se encuentra en fase de aprendizaje. En inp_layer le pasamos la imagen 
# del gato o la cerveza.
# Las salida seran grad y loss.
optimize_gradient = K.function([inp_layer, K.learning_phase()], [grad, loss])

# Ahora itero
adv = np.copy(x)  # Hago una copia de la imagen asi no altero la original.

# Para que la imagen final no se vea muy rara al ojo humano.
# limitamos la perturbacion que la misma puede tener.
pert = 0.01
max_pert = x + pert
min_pert = x - pert

cost = 0.0
while cost < 0.95:
    gr, cost = optimize_gradient([adv, 0])
    adv += gr  # Sumo a los pixeles de la imagen los valores del gradiente.
    adv = np.clip(adv, min_pert, max_pert)  # Limit the values in an array.
    adv = np.clip(adv, -1, 1)  # Limit the values in an array.
    print('Target cost: ', cost)

# Ploteo la imagen alterada adv. Antes debo pasar los pixeles nuevamente de 
# 0 a 255 haciendo los pasos contrarios
adv /= 2
adv += 0.5
adv *= 255
im = Image.fromarray(adv[0].astype(np.uint8))
im.save('hacked.png')
plt.imshow(adv[0].astype(np.uint8))
plt.show()

# Finalmente si paso la imagen hacked.png como entrada la detecta como un limon