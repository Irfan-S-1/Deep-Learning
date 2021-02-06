# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:49:11 2021

@author: Irfan Sheikh
"""

import pandas as pd
import numpy as np
import tensorflow as tf
tf.__version__

df=pd.read_excel("D:\Data Science\Deep Learning\Deep Learning Practice\ANN\\Folds5x2_pp.xlsx")
 
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


#Initializing ANN#

ann=tf.keras.models.Sequential()

#Adding Input layer and first hidden layer#

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

#Adding Second hidden layer#
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

#Adding outpu layer#
ann.add(tf.keras.layers.Dense(units=1))

#Compiling ANN#
ann.compile(optimizer="adam",loss="mean_squared_error")

#Training ANN 

ann.fit(X_train,Y_train,batch_size=32,epochs=100)

#Predicting on test data#

y_pred=ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))
