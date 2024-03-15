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

def hashing_and_splitting(adj_close_df):
    # calculate a checksum of unique identifier for the row (here we use index)
    checksum = np.array([crc32(v) for v in adj_close_df.index.values])
    # fraction of data to use for testing only
    test_ratio = 0.2
    # which rows should be in test; scale ratio by maximum checksum value: 2 to power 32
    test_indices = checksum < 0.2 * 2**32
    # split into two dataframes
    train_df = adj_close_df[~test_indices] # all rows where test_indices is NOT true
    test_df = adj_close_df[test_indices]  # all rows for test_indices
    return train_df, test_df

def xtrain_ytrain(adj_close_df):
    train_set, test_set = hashing_and_splitting(adj_close_df)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(train_set)
    X_train = []
    y_train = []
    for i in range(60, len(adj_close_df)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

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

def visualizing(xtrain, ytrain, xtest):
    predicted_price, real_price = predicted_price(xtrain, ytrain, xtest)
    #Visualizing the prediction
    plt.figure()
    plt.plot(real_price, color = 'r', label = 'Close')
    plt.plot(predicted_price, color = 'b', label = 'Prediction')
    plt.xlabel('Date')
    plt.legend()
    plt.show()