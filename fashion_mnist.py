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

# make model
model = Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(512, activation='sigmoid'))
model.add(layers.Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(classNum, activation='softmax'))

#compile model
model.compile(loss = categorical_crossentropy,
              optimizer = Adam(),
              metrics=['accuracy'])

# define some hyper parameters and train
#es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model.fit(train_images, train_labels, validation_split  = 0.2, batch_size = 128, epochs=10) 
loss, score = model.evaluate(test_images, test_labels, verbose = 1)
print("Test accuracy: ", score)
