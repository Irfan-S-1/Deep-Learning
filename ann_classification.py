# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:37:53 2021

@author: Irfan Sheikh
"""
#Importing Tensorflow 2.4#
import pandas as pd
import numpy as np
import tensorflow as tf

#Importing dataset#
df=pd.read_csv("D:\\Data Science\\Deep Learning\\Deep Learning Practice\\Churn_Modelling.csv")
#Dropping unnecessary columns#
df=df.drop(["RowNumber","CustomerId","Surname"],axis=1)

#Checking Null Values#
count=df.isnull().sum()

#Splitting data into X and Y#
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

#Converting categorical variables into Discrete#
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X["Gender"]=le.fit_transform(X["Gender"])
print(X)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X=np.array(ct.fit_transform(X))

#Splitting data into Train,test#
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#Standardizing data#
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()

X_train=SC.fit_transform(X_train)
X_test=SC.fit_transform(X_test)

#Building ANN with input layer,hidden layer and output layer#
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))


ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


ann.fit(X_train,Y_train,batch_size=32,epochs=100)

#Testing our Model#
print(ann.predict(SC.transform([[1,0,0,600,1,40,3,6000,2,1,1,50000]]))>0.5)

Y_pred=ann.predict(X_test)
Y_pred=(Y_pred>0.5)

#Creating confusion matrix and accuracy score#
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_pred,Y_test)
print(cm)
accuracy_score(Y_test,Y_pred)
