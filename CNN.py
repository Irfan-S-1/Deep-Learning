# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:37:35 2021

@author: Irfan Sheikh
"""
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
os.chdir('D:\Data Science\Deep Learning\Deep Learning Practice\CNN')

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_datagen=ImageDataGenerator(rescale =1./255)

test_set=test_datagen.flow_from_directory("dataset/test_set",target_size=(64,64),batch_size=32,class_mode="binary")

#Initializing CNN%

cnn=tf.keras.models.Sequential()

#Convolution layer#
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,3]))

#Pooling#
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Adding Second Convolutional layer with max pooling#

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Flattening

cnn.add(tf.keras.layers.Flatten())

#Full Connection#

cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))

#Output layer#

cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

#Compiling CNN#

cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Testing CNN on training and test set#

cnn.fit(x=training_set, validation_data=test_set ,epochs=25)


#Making a Prediction on our model#
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_5.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] > 0.5:
  prediction = 'dog'
else:
  prediction = 'cat'
    


    
print(prediction)