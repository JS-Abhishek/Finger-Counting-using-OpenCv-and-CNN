# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:20:59 2020

@author: jsabh
"""



#training the data using CNN
import tensorflow as tf
from tensorflow import keras

classifier = keras.models.Sequential()

classifier.add(keras.layers.Conv2D(32,(3,3),activation = "relu", input_shape = (64,64,1)))
classifier.add(keras.layers.MaxPooling2D((2,2)))
classifier.add(keras.layers.Conv2D(64,(3,3),activation = "relu"))
classifier.add(keras.layers.Conv2D(64,(3,3),activation = "relu"))
classifier.add(keras.layers.MaxPooling2D((2,2)))
classifier.add(keras.layers.Conv2D(128,(3,3),activation = "relu"))
classifier.add(keras.layers.MaxPooling2D((2,2)))
classifier.add(keras.layers.Conv2D(256,(3,3),activation = "relu"))
classifier.add(keras.layers.MaxPooling2D((2,2)))
classifier.add(keras.layers.Flatten())
classifier.add(keras.layers.Dense(150,activation = "relu"))
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(6,activation = "softmax"))

classifier.compile(optimizer = 'adam', loss = "categorical_crossentropy",metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(
    rescale = 1./255)

training_set = train_datagen.flow_from_directory('images/train', target_size = (64,64),
                                                 batch_size = 32, color_mode = 'grayscale',
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('images/test',target_size = (64,64),
                                            batch_size = 32, color_mode = 'grayscale',
                                            class_mode = 'categorical')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/GPU:0'):
    history = classifier.fit_generator(
        training_set,steps_per_epoch = 64, epochs = 10, 
        validation_data = test_set, validation_steps = 28)


model_json = classifier.to_json()
with open("model-bw-hand.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw-hand.h5')


