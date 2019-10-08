"""Academical examples on image classifications.

This module is an experiment for classifying images. The goal is to train a
model that can successfully identify images annotated in certain way. The
annotation is a small image that's super-imposed on a background image.
"""
import keras
import numpy
from keras import layers
from keras import models
from keras import optimizers
from matplotlib import pyplot as plt

from qpylib import logging
from qpylib import t

IMAGE_SIZE = 128

# Image array is a 3d float array; its 1st dimension is y and 2nd dimension
# is x, and 3rd dimension has size 1 and it is the grey scale. This convention
# is chosen to match pyplot.imshow.
Image = numpy.array


def CreateBlankImage(
    height: int,
    width: int,
) -> Image:
  """Creates a blank (0) image."""
  return numpy.zeros((width, height, 1), dtype=float)


def CreateBlankImageWithNoise(
    height: int,
    width: int,
    noise_strength: float = 0.1,
) -> Image:
  """Creates a almost blank (0) image."""
  return numpy.random.uniform(0.0, noise_strength, (width, height, 1))


def AddBoxToImage(img: Image, y: int, x: int, size: int = 5):
  """Adds a box (color 1) to the image."""
  img[y:y + size, x:x + size] = 1.0


def CreateImageData(
    num_annotated_images: int,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> t.Tuple[numpy.array, numpy.array]:
  """Creates an image data set of given size.

  Args:
    num_annotated_images: number of annotated images.
    height: image height.
    width: image width.

  Returns:
    A tuple of images and labels (one-hot representation), order is randomized.
  """
  images_and_labels = []
  for _ in range(num_annotated_images):
    y = numpy.random.randint(0, height - 1)
    x = numpy.random.randint(0, width - 1)
    img = CreateBlankImageWithNoise(height, width)
    AddBoxToImage(img, y=y, x=x)
    images_and_labels.append((img, numpy.array([x, y])))

  numpy.random.shuffle(images_and_labels)
  return (
    numpy.array([e[0] for e in images_and_labels]),
    numpy.array([e[1] for e in images_and_labels]))


def CreateDefaultOptimizer() -> optimizers.Optimizer:
  """Creates a default optimizer."""
  # Ref:
  #   https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
  return optimizers.RMSprop(lr=0.00025)


def CreateConvolutionModel(
    image_shape: t.Tuple[int, int, int],
    activation: t.Text = 'relu',
    optimizer: optimizers.Optimizer = None,
) -> keras.Model:
  """Creates a convolution model suitable for screen based learning."""
  if optimizer is None:
    optimizer = CreateDefaultOptimizer()

  model = models.Sequential()
  model.add(layers.Conv2D(
    16,
    kernel_size=4,
    activation=activation,
    input_shape=image_shape,
  ))
  model.add(layers.Conv2D(
    16,
    kernel_size=4,
    activation=activation))
  model.add(layers.Flatten())
  model.add(layers.Dense(units=6, activation=activation))
  model.add(layers.Dense(units=6, activation=activation))
  model.add(layers.Dense(units=2))

  model.compile(loss='mse', optimizer=optimizer)

  return model


def PlotImage(img: Image):
  y_size, x_size, _ = img.shape
  plt.imshow(img.reshape(y_size, x_size))
  plt.show()


def MainTest():
  images, labels = CreateImageData(num_annotated_images=5)
  for idx in range(5):
    logging.printf('Label %d: %s', idx, labels[idx])
    PlotImage(images[idx])


def MainTrain():
  images, labels = CreateImageData(
    num_annotated_images=50000,
  )
  model = CreateConvolutionModel((IMAGE_SIZE, IMAGE_SIZE, 1))
  model.fit(images, labels)

  images, labels = CreateImageData(
    num_annotated_images=50,
  )
  predicted_labels = model.predict(images)
  print('Error: %s' % numpy.abs(predicted_labels - labels).mean(axis=0))


if __name__ == '__main__':
  # MainTest()
  MainTrain()
