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
    prices = si.get_data(ticker, start_date = start_date, end_date = end_date)
    stocks_dict[ticker] = prices['adjclose']
  
  stocks_prices = pd.DataFrame.from_dict(stocks_dict)
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