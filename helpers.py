#libraries
import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yahoo_fin.stock_info as si
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
    split_index = int((len(adj_close_df))*0.80)
    #train_set, test_set = hashing_and_splitting(adj_close_df)
    train_set = pd.DataFrame(adj_close_df.iloc[0:split_index])
    test_set = pd.DataFrame(adj_close_df.iloc[split_index:])
    
    sc = MinMaxScaler(feature_range = (0, 1))
    sc.fit(train_set)
    training_set_scaled = sc.fit_transform(train_set)
    test_set_scaled = sc.transform(test_set)
        
    xtrain = []
    ytrain = []
    for i in range(60, training_set_scaled.shape[0]):
        xtrain.append(training_set_scaled[i-60:i, 0])
        ytrain.append(training_set_scaled[i, 0]) 
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    
    past_100_days = train_set.tail(100) #testing on the last 100 days of the train dataset
    test_set = pd.concat([past_100_days, train_set], ignore_index=True)
    xtest = []
    ytest = []
    for i in range(20, test_set_scaled.shape[0]):
        xtest.append(test_set_scaled[i-20:i,0])
        ytest.append(test_set_scaled[i,0])
    xtest, ytest = np.array(xtest), np.array(ytest)
    return xtrain, ytrain, xtest, ytest, sc

def lstm_model(xtrain, ytrain):
    model = Sequential() 
    model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (xtrain.shape[1], 1)))
    model.add(Dropout(0.2)) 
    model.add(LSTM(units = 60, activation='relu', return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(units = 80, activation = 'relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, ytrain, epochs = 20, batch_size=32, verbose=1)
    return model

def predictions(model, xtest, ytest, sc):
    #Making predictions on the test data
    predicted_stock_sc = model.predict(xtest)
    #predicted_stock_real = sc.inverse_transform(predicted_stock_sc)
    #rescale the predicted and original labels
    scale = 1/sc.scale_  # converts it back to normal price
    predicted_stock_sc = predicted_stock_sc * scale
    ytest = ytest * scale
    return predicted_stock_sc, ytest

###plotting 
def visualizing(model, xtest, ytest, sc):
    predicted_stock, yt = predictions(model, xtest, ytest, sc)
    #Visualizing the prediction
    plt.figure()
    plt.plot(yt, color = 'r', label = 'Closing price')
    plt.plot( predicted_stock, color = 'b', label = 'Prediction')
    plt.xlabel('Date')
    plt.legend()
    plt.show()