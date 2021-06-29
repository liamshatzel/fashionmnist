# -*- coding: utf-8 -*-
"""Fashion MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uNOsFVqab1G2jQGoAf3GKjxOKg9t6Ilh
"""

# install packages
!pip install keras
!pip install tensorflow

# import and download data
from keras.datasets import fashion_mnist
import numpy as np
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#preprocessing
train_images = np.expand_dims(train_images.astype('float16') / 255., axis=3)
test_images = np.expand_dims(test_images.astype('float16') / 255., axis=3)


# import some stuff
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, Input, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers

#reformat labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# define number of classes 
classNum = 10

#Augment Data:
data_augmentation = keras.Sequential([ 
  #layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28, 28, 1)),
  #layers.experimental.preprocessing.RandomRotation(0.1),
  #layers.experimental.preprocessing.RandomZoom(0.1),
      
])

# make model
model = Sequential()
model.add(data_augmentation)
model.add(Flatten(input_shape = (28, 28)))
#model.add(Dense(512, kernel_regularizer = tf.keras.regularizers.l2(0.0001), activation='elu'))
model.add(Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(Dense(256, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(classNum, activation='softmax'))


#TODO
'''
Try 60 epochs
Take out regularizer
Change layer sizes
how to fix overfit problem
Still overfit
reduce layer numbers and size
add image distortion
'''


#compile model
model.compile(loss = categorical_crossentropy,
              optimizer = Adam(),
              metrics=['accuracy'])

# define some hyper parameters and train
model.fit(train_images, train_labels, batch_size = 128, epochs = 40)
loss, score = model.evaluate(test_images, test_labels, verbose = 1)
print("Test accuracy: ", score)