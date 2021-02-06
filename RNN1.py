# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:12:37 2021

@author: Irfan Sheikh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("D:\Data Science\Deep Learning\Deep Learning Practice\RNN")


df_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=df_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)


X_train=[]
Y_train=[]
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])
X_train,Y_train=np.array(X_train),np.array(Y_train)

X_train=np.reshape(X_train ,(X_train.shape[0],X_train.shape[1],1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor=Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))

# Adding the output layer
regressor.add(Dense(units = 1))


#Compiling RNN#
regressor.compile(optimizer="adam",loss="mean_squared_error")

#Fitting data to our model#
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)


#Importing test dataset#
df_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_prices=df_test.iloc[:,1:2].values

dataset_total=pd.concat([df_train["Open"],df_test["Open"]] ,axis = 0)
inputs=dataset_total[len(dataset_total)-len(df_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

X_test=[]

for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
    
X_test=np.array(X_test)

X_test=np.reshape(X_test ,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price=regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(real_stock_prices,color="red",label="Real Google Stock Prices")
plt.plot(predicted_stock_price,color="blue",label="Predicted Stock Prices")
plt.title("Google Stock Price Prediction")
plt.xlabel("time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
