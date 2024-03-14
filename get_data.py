import yahoo_fin.stock_info as si
import pandas as pd
import datetime

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
  