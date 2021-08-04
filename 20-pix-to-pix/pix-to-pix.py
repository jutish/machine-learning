import tensorflow as tf
import os
import time
import datetime
import matplotlib.pyplot as plt

# # Load the dataset
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
print(PATH,'\n',path_to_zip)
# # Each original image is of size 256 x 512 containing two 256 x 256 images:
# sample_image = tf.io.read_file(PATH + 'train/1.jpg')
# sample_image = tf.io.decode_jpeg(sample_image)
# # print(sample_image)
# # plt.figure()
# # plt.imshow(sample_image)
# # plt.show()

# # You need to separate real building facade images from the architecture 
# # label images—all of which will be of size 256 x 256.
# # Define a function that loads image files and outputs two image tensors:
def load(image_file):
    # Read and decode an image to uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

# # Plot a sample of the input (architecture label image) and real
# # (building facade photo) images:
inp, re = load(PATH + 'train/18.jpg')
# # Casting to int for matplotlib to display the images
# # plt.figure()
# # plt.imshow(inp / 255.0)
# # plt.figure()
# # plt.imshow(re / 255.0)
# # plt.show()

# # As described in the pix2pix paper, you need to apply random jittering and
# # mirroring to preprocess the training set.

# # Define several functions that:

# # Resize each 256 x 256 image to a larger height and width -> 286 x 286.
# # Randomly crop it back to 256 x 256.
# # Randomly flip the image horizontally i.e. left to right (random mirroring).
# # Normalize the images to the [-1, 1] range.

# # The facade training set consist of 400 images
BUFFER_SIZE = 400
# # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# # Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


# # Resize each 256 x 256 image to a larger height and width—286 x 286.
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image


# Randomly crop it back to 256 x 256.
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image[0], cropped_image[1]


# # Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image


# Defining jitter function that use the other ones
@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)
  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image


# # plt.figure(figsize=(6, 6))
# # for i in range(4):
# #   rj_inp, rj_re = random_jitter(inp, re)
# #   plt.subplot(2, 2, i + 1)
# #   plt.imshow(rj_re / 255.0)
# #   plt.axis('off')
# # plt.show()

# # Having checked that the loading and preprocessing works, let's define a
# # couple of helper functions that load and preprocess the training and test
# # sets:

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

# # Build an input pipeline with tf.data
train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

# for input_image, real_image in train_dataset.take(5):
#   plt.imshow(((input_image[0,...]) + 1 )/ 2)
#   plt.show()

# Comienzo a armar la arquitectura de mi red
# Del Paper tengo que la red esta formada por un enconder y un decoder (U-net)
# y un discriminador (patch-gan)
# El enconder tiene la siguiente arquitectura: C64-C128-C256-C512-C512-C512
# El decoder tiene la siguiente arquitectura: CD512-CD512-CD512-C512-C256-C128-C64
# El discriminidor la siguiente arquitectura: ??
# Ck implica un bloque formado por 3 capas (Convolution-BatchNorm-ReLU) donde
# k indica el numero de filtros de la capa Convolutional.
# CDk implica otro bloque formado por 4 capas (Convolution-BatchNorm-
# Dropout-ReLU) donde k indica el numero de filtros de la capa convolutional.
# En resumen la arquitectura es un Encoder con 6 bloques C cada donde la capa convolutional tiene k filtros.
# y el decoder son 2 Bloques tipo CD y 5 bloques tipo C cada una con k filtros en la capa convolutional
# Tambien consta de un discrimiandor del tipo Patch-Gan
# A la arquitectura dada hay que setearles algunas excepciones que se explican
# en el vídeo: https://youtu.be/YsrMGcgfETY Minuto: 52:00

from tensorflow.keras import *
from tensorflow.keras.layers import *


# Se encarga del Encoder
def downsample(filters, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0, 0.02)  # Los valores iniciales deben ser normales con una SD de 0.02
  result = Sequential()  # Especifico a keras que lo siguiente es una secuencia de capas
  # Capa de convolucion
  result.add(Conv2D(filters,
                    kernel_size = 4,  # 4x4
                    strides=2,  # Movimiento del filtro 2,
                    padding='same',
                    kernel_initializer=initializer,
                    use_bias=not apply_batchnorm))  # Usa bias siempre que no se use la capa de batch_norm
  # Capa de batch normalization
  if apply_batchnorm:
    result.add(BatchNormalization())
  # Capa de activacion basa en Leaky Relu un version suavisada de ReLU
  result.add(LeakyReLU())  
  return result

## Test downsample
# down_model = downsample(6)
# down_result = down_model(tf.expand_dims(inp, 0))
# tf.expand_dims(inp, 0).shape.as_list()
# print(down_result.shape)


# Se encarga del Decoder
def upsample(filters, apply_dropout=True):
  initializer = tf.random_normal_initializer(0, 0.02)  # Los valores iniciales deben ser normales con una SD de 0.02
  result = Sequential()  # Especifico a keras que lo siguiente es una secuencia de capas
  # Capa de convolucion
  result.add(Conv2DTranspose(filters,
                             kernel_size = 4,  # 4x4
                             strides=2,  # Movimiento del filtro 2,
                             padding='same',
                             kernel_initializer=initializer,
                             use_bias=False))
  # Capa de batch normalization
  result.add(BatchNormalization())
  # Capa de dropout
  if apply_dropout:
    result.add(Dropout(0.5))
  # Capa de activacion basa en Relu
  result.add(ReLU())
  return result


## Test upsample
# up_model = upsample(3)
# up_result = up_model(down_result)
# print (up_result.shape)


# Defino el generador de la arquitectura
def generator():
  # Input
  inputs = tf.keras.layers.Input(shape=[256, 256, 3]) # Imagenes de MxN pixeles con 3 de profundidad
  # Encoder
  down_stack = [
    downsample(64, apply_batchnorm=False),  # (bs, 128, 128, 64) bs=batch_size
    downsample(128),                        # (bs, 64, 64, 128)
    downsample(256),                        # (bs, 32, 32, 256)
    downsample(512),                        # (bs, 16, 16, 512)
    downsample(512),                        # (bs, 8, 8, 512)
    downsample(512),                        # (bs, 4, 4, 512)
    downsample(512),                        # (bs, 2, 4, 512)
    downsample(512)                         # (bs, 1, 1, 512)
  ]
  # Decoder
  up_stack = [
    upsample(512),                          # (bs, 2, 2, 1024)
    upsample(512),                          # (bs, 4, 4, 1024) 
    upsample(512),                          # (bs, 8, 8, 1024) 
    upsample(512, apply_dropout=False),     # (bs, 16, 16, 1024)
    upsample(256, apply_dropout=False),     # (bs, 32, 32, 512)
    upsample(128, apply_dropout=False),     # (bs, 64, 65, 256)
    upsample(64, apply_dropout=False)       # (bs, 128, 128, 128)
  ]
  # Capa de salida
  initializer = tf.random_normal_initializer(0, 0.02)
  last = Conv2DTranspose(filters=3,   # 3 canales de color de la imagen, por eso tres filtros
                         kernel_size=4,  # 
                         strides=2,  # Duplica el tamañoa anterior de 128x128 a 256x256
                         padding='same',
                         kernel_initializer=initializer,
                         activation='tanh')  # La salida es de -1 a 1 por eso usamos tanh (bs, 256, 256, 3)
  # Conecto todas las capas donde la salida de una es la entrada de la otra
  x = inputs  # inputs es la primera entrada, la imagen.
  s = list()  # Uso este vector para guardar la salida de cada capa, esto par poder aplicar skipconnections.
  # Trabajo con el enconder
  for down in down_stack:
    x = down(x)
    s.append(x)
  s = reversed(s[:-1])  # El ultimo elemento no es necesario y lo saco, ademas da vuelta la lista ya que por ej. la capa 1 del decoder debe conectarse a la penultima del enconder.
  # Trabajo con el decoder
  for up, sk in zip(up_stack, s):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, sk])  # Concateno el resultado con la skiconnection corrsp. y eso es la entrada a la siguiente iteracion.
  # Paso por la ultima capa
  x = last(x)
  # Retorno un modelo
  return Model(inputs=inputs, outputs=x)

# # Imprimo la arquitectura del generador
generator = generator()
# tf.keras.utils.plot_model(generator, to_file='generator.png', show_shapes=True)

# # Pruebo el generador
# gen_output = generator(inp[tf.newaxis, ...], training=False)
# plt.imshow(gen_output[0, ...])
# plt.show()


# Es una arquitectura similar a la parte del downsample() del generador y se llama PatchGan.
def discriminator():
  ini = Input(shape=[256, 256, 3], name='input_img')  # Puede ser la imagen real o bien el input del generador.
  gen = Input(shape=[256, 256, 3], name='gen_img')  # Imagen generada
  con = tf.keras.layers.Concatenate()([ini, gen])
  initializer = tf.random_normal_initializer(0, 0.02)
  down1 = downsample(64, apply_batchnorm=False)(con)
  down2 = downsample(128)(down1)
  down3 = downsample(256)(down2)
  down4 = downsample(512)(down3)
  last = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=4,
                                strides=1,
                                kernel_initializer=initializer,
                                # activation='sigmoid',  # Esta linea esta en el paper, pero no aparece ni en el video y tampoco en la docu de TensorFlow, yo la pongo porque tengo huevos..!!
                                padding='same')(down4)
  return tf.keras.Model(inputs=[ini, gen], outputs=last)

# # Imprimo la arquitectura del discriminador
discriminator = discriminator()
# tf.keras.utils.plot_model(discriminator, to_file='discriminator.png', show_shapes=True)

# # Imprimo la salida del descriminador
# disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()
# plt.show()


# Funcion de coste del discriminador
# from_logits = True pone los valores de la imagen entre 0 y 1 que seria pasarla por una sigmoide 
# como lo pedia el paper en el discriminador.
# La funcion de coste va a recibir la salida del discriminador cuando ve una imagen real
# y la salida del discriminador cuando ve una imagen generada, es decir recibe 2 patchGans
# En base a esos dos patchGans es que calcula real_loss y generated_loss
# luego la suma de ambos es total_disc_loss que si es bajo significa que el discriminador
# esta haciendo bien su trabajo.
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) 
def discriminator_loss(disc_real_output, disc_generated_output):
  # diferencia entre los True por ser real y el detectado por el discriminador
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss


# Funcion de coste del generador
LAMBDA = 100
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  #mean_absoulte_error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss


# Checkpoints por si se clava el sistema para poder restaurar desde el ultimo epoch
import os
# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
# checkpoint.restore(tf.train.latest_checkpoint(CKPATH)).assert_consumed()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Write a function to plot some images during training.
# Pass images from the test set to the generator.
# The generator will then translate the input image into the output.
# The last step is to plot the predictions and voila!
def generate_images(model, test_input, tar, save_filename=False, display_imgs=True):
  prediction = model(test_input, training=True)
  # if save_filename:
  #   tf.keras.preprocessing.image.save_img(PATH + '/output/' + save_filename + '.jpg' + prediction[0, ...])
  plt.figure(figsize=(10,10))
  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  if display_imgs:
    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # getting the pixel values between [0, 1] to plot it
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
  plt.show()

# Test the function
# for example_input, example_target in test_dataset.take(1):
#   generate_images(generator, example_input, example_target)


@tf.function()
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
    output_image = generator(input_image, training=True)
    output_gen_discr = discriminator([output_image, input_image], training=True)
    output_trg_discr = discriminator([target, input_image], training=True)
    discr_loss = discriminator_loss(output_trg_discr, output_gen_discr)
    gen_loss = generator_loss(output_gen_discr, output_image, target)
    generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

# Definimos como se va a entrenar el modelo
def train(dataset, epochs):
  for epoch in range(epochs):
    imgi = 0
    for input_image, target in dataset:
      print('epoch: ' + str(epoch) + ' - train:' + str(imgi))
      imgi += 1
      train_step(input_image, target) # Aca se entrena
      # clear_output(wait=True)
    imgi = 0
    for inp, tar in test_dataset.take(5):
     # Uso el modelo entrenado en 5 imagenes del dataset de test
     generate_images(generator, inp, tar, str(imgi) + '_' + str(epoch), display_imgs=True)
     imgi += 1
    #saving checkpoint every 25 epochs
    if (epoch + 1) % 25 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

# # Ejecutamos

train(train_dataset, 1)