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
from keras.callbacks import History
from zlib import crc32
import os
import pickle
import yfinance as yf

history = History()

###pickling, both dumping and loading
def pickle_dump(stocks_prices):
    with open("stocks_prices_dataframe.pkl", "wb") as f:
        pickle.dump(stocks_prices, f)

def pickle_load(filename):
    with open(filename, "rb") as f:
        stocks_prices = pickle.load(f)
    return stocks_prices

#load data, online or from pickle
def load_dataframe(years):
    #check if pickle file exists
    if(os.path.isfile("stocks_prices_dataframe.pkl")):
        stock_prices = pickle_load("stocks_prices_dataframe.pkl")
        tickers = stock_prices.columns.tolist()
    #if pickle doesn't exist, then pick data online
    else:
        tickers = get_nasdaq_tickers()
        stock_prices = loaded_df(years=years, tickers=tickers)

    return stock_prices, tickers

###hashing
def hashing_and_splitting(adj_close_df):
    # calculate checksum for every index
    checksum = np.array([crc32(v) for v in adj_close_df.index.values])
    # fraction of data to use for testing only
    test_ratio = 0.2
    # pick 20% of indices for testing
    test_indices = checksum < test_ratio * 2**32

    return adj_close_df[~test_indices], adj_close_df[test_indices]


###getting data online 
def get_nasdaq_tickers():
  # get NASDAQ ticker
  tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
  df = tables[4]
  # clean df
  df.drop(['Company','GICS Sector', 'GICS Sub-Industry'], axis=1, inplace=True)
  # put data in a list
  tickers = df['Ticker'].values.tolist()
  return tickers

def loaded_df(years, tickers):
  stocks_dict = {}
  # set the period
  time_window = 365*years
  start_date = dt.datetime.now() - dt.timedelta(time_window)
  end_date = dt.datetime.now()

  # get data and load into a dataframe
  for i, ticker in enumerate(tickers):
    print('Getting {} ({}/{})'.format(ticker, i, len(tickers)))
    prices = yf.download(ticker, start = start_date, end = end_date)
    stocks_dict[ticker] = prices['Adj Close']
  
  stocks_prices = pd.DataFrame.from_dict(stocks_dict)
  pickle_dump(stocks_prices=stocks_prices)
  return stocks_prices


###data cleaning
def clean_df(percentage, tickers, stocks_prices):
    if percentage > 1: 
        percentage = percentage/100
    # check NaN values
    for ticker in tickers:
        nan_values = stocks_prices[ticker].isnull().values.any()
        if nan_values == True:
            # count NaN values
            count_nan = stocks_prices[ticker].isnull().sum()
            # remove NaN values
            if count_nan > (len(stocks_prices)*percentage):
                stocks_prices.drop(ticker, axis=1, inplace=True)
    return stocks_prices



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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])  # Add metrics=['accuracy'] to monitor accuracy
    model.fit(xtrain, ytrain, epochs=100, batch_size=32, verbose=1)
    return model


def predictions(model, xtest, ytest, sc):
    #Making predictions on the test data
    predicted_stock_sc = model.predict(xtest)
    #predicted_stock_real = sc.inverse_transform(predicted_stock_sc)
    predicted_stock_sc= predicted_stock_sc.reshape(-1, 1)
    predicted_stock = sc.inverse_transform(predicted_stock_sc)
    ytest = ytest.reshape(-1, 1)
    ytest = sc.inverse_transform(ytest)
    return predicted_stock, ytest

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
    
def model_data(model):
    #part of code pertaining plotting loss and accuracy of the model
    print(history.history.keys())
    train_loss = history.history['loss']
    xc = range(1, 101)  
    plt.figure(figsize=(12, 6))

    #plotting aoss
    plt.subplot(1, 2, 1)
    plt.plot(xc, train_loss, '-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()