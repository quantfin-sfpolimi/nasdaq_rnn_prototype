import yahoo_fin.stock_info as si
import pandas as pd
import datetime
import pickle5 as pickle

# get NASDAQ ticker
tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
df = tables[4]

# clean df
df.drop(['Company','GICS Sector', 'GICS Sub-Industry'], axis=1, inplace=True)

# put data in a list
tickers = df['Ticker'].values.tolist()

# create an empty data frame
stocks_prices = pd.DataFrame()

# set the period
time_window = 365*5
start_date = datetime.datetime.now() - datetime.timedelta(time_window)
end_date = datetime.datetime.now()

# get data and load into a dataframe
for ticker in tickers:
  stock_prices = si.get_data(ticker, start_date = start_date, end_date = end_date)
  stocks_prices[ticker] = stock_prices['adjclose']

# check NaN values
for ticker in tickers:
  nan_values = stocks_prices[ticker].isnull().values.any()
  if nan_values == True:
    # count NaN values
    count_nan = stocks_prices[ticker].isnull().sum()
    # remove NaN values
    if count_nan > (len(stocks_prices)*0.1):
      stocks_prices.drop(ticker, axis=1, inplace=True)

# clean remaining NaN
stocks_prices.dropna(inplace = True)

# saving stocks_prices in a pickle file
with open("stocks_prices_dataframe.pkl", "wb") as f:
    pickle.dump(stocks_prices, f)