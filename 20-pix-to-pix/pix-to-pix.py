import tensorflow as tf
import os
import time
import datetime
import matplotlib.pyplot as plt

# Load the dataset
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
print(PATH)
# Each original image is of size 256 x 512 containing two 256 x 256 images:
sample_image = tf.io.read_file(PATH + 'train/1.jpg')
sample_image = tf.io.decode_jpeg(sample_image)
# plt.figure()
# plt.imshow(sample_image)
# plt.show()

# You need to separate real building facade images from the architecture 
# label images—all of which will be of size 256 x 256.
# Define a function that loads image files and outputs two image tensors:
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

# Plot a sample of the input (architecture label image) and real
# (building facade photo) images:
inp, re = load(PATH + 'train/100.jpg')
# Casting to int for matplotlib to display the images
# plt.figure()
# plt.imshow(inp / 255.0)
# plt.figure()
# plt.imshow(re / 255.0)
# plt.show()

# As described in the pix2pix paper, you need to apply random jittering and
# mirroring to preprocess the training set.

# Define several functions that:

# Resize each 256 x 256 image to a larger height and width—286 x 286.
# Randomly crop it back to 256 x 256.
# Randomly flip the image horizontally i.e. left to right (random mirroring).
# Normalize the images to the [-1, 1] range.

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
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

# plt.figure(figsize=(6, 6))
# for i in range(4):
#   rj_inp, rj_re = random_jitter(inp, re)
#   plt.subplot(2, 2, i + 1)
#   plt.imshow(rj_inp / 255.0)
#   plt.axis('off')
# plt.show()

# Having checked that the loading and preprocessing works, let's define a
# couple of helper functions that load and preprocess the training and test
# sets:

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

# Build an input pipeline with tf.data
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
# Del Paper tengo que la red esta formada por un enconder y un decoder
# El enconder tiene la siguiente arquitectura: C64-C128-C256-C512-C512-C512
# El decoder tiene la siguiente arquitectura: CD512-CD512-CD512-C512-C256-C128-C64
# Ck implica una capa superior formado por k filtros donde cada filtro consta de 3 capas basicas
# una capa Convolution, otra de Batch Normalization y una de ReLu
# CDk implica capa superior formada por k filtros donde cada filtro consta de 4 capas
# una capa Convolution, otra de Batch Normalization, una de Dropout y una de ReLu
# En resumen la arquitectura es un Encoder con 6 Capas superiores tipo C cada una con k filtros
# y el decoder son 2 Capas Superiores tipo CD y 5 Capas sup. tipo C cada una con k filtros dentro
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


