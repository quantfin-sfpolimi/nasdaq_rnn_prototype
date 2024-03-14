import yahoo_fin.stock_info as si
import pandas as pd
import datetime
import pickle

#make function to get list of tickers
def get_nasdaq_tickers():
  # get NASDAQ ticker
  tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
  df = tables[4]
  # clean df
  df.drop(['Company','GICS Sector', 'GICS Sub-Industry'], axis=1, inplace=True)
  # put data in a list
  tickers = df['Ticker'].values.tolist()
  return tickers

#function to get data from a timeframe and load it into a dataframe
def loaded_df(years, tickers):
  stocks_dict = {}
  # set the period
  time_window = 365*years
  start_date = datetime.datetime.now() - datetime.timedelta(time_window)
  end_date = datetime.datetime.now()

  # get data and load into a dataframe
  for i, ticker in enumerate(tickers):
    print('Getting {} ({}/{})'.format(ticker, i, len(tickers)))
    prices = si.get_data(ticker, start_date = start_date, end_date = end_date)
    stocks_dict[ticker] = prices['adjclose']
  
  stocks_prices = pd.DataFrame.from_dict(stocks_dict)
  return stocks_prices

#dataframe to drop collumns with too much dirty date (under certain given percentage)
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
  # clean remaining NaN
  stocks_prices.dropna(inplace = True)
  return stocks_prices


#example of usage
tickers = get_nasdaq_tickers()
stocks_prices = loaded_df(5, tickers=tickers)
stocks_prices = clean_df(0.1, tickers=tickers, stocks_prices=stocks_prices)
# saving stocks_prices in a pickle file
with open("stocks_prices_dataframe.pkl", "wb") as f:
    pickle.dump(stocks_prices, f)
    
    