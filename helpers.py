#libraries
import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from zlib import crc32

###pickling, hashing, getting data online 


###data cleaning


### rnn model
def xtrain_ytrain(adj_close_df):
    train_set, test_set = hashing_and_splitting(adj_close_df)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(train_set)
    xtrain = []
    ytrain = []
    for i in range(60, len(adj_close_df)):
        xtrain.append(training_set_scaled[i-60:i, 0])
        ytrain.append(training_set_scaled[i, 0]) 
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    return xtrain, ytrain, test_set

def lstm_model(xtrain, ytrain):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    #compiling and fitting the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(xtrain, ytrain, epochs = 15, batch_size = 32)
    return model

def predictions(xtrain, ytrain, xtest):
    #Making predictions on the test data
    sc = MinMaxScaler(feature_range = (0, 1))
    predicted_stock_price = lstm_model(xtrain, ytrain).predict(xtest)
    real_stock_price = sc.inverse_transform(predicted_stock_price)
    return predicted_stock_price, real_stock_price

###plotting 
def visualizing(xtrain, ytrain, xtest):
    predicted_price, real_price = predictions(xtrain, ytrain, xtest)
    #Visualizing the prediction
    plt.figure()
    plt.plot(real_price, color = 'r', label = 'Close')
    plt.plot(predicted_price, color = 'b', label = 'Prediction')
    plt.xlabel('Date')
    plt.legend()
    plt.show()