#import libraries
from helpers import xtrain_ytrain, visualizing, load_dataframe, clean_df, lstm_model
import yfinance as yf
import pandas as pd

#load data, clean data frame (closing stock prices)
stocks_prices, tickers = load_dataframe(years=10)
clean_df(10, tickers=tickers, stocks_prices=stocks_prices)
# clean remaining NaN
stocks_prices.dropna(inplace = True)

# get one stock, only for prototype purpose (ADBE)
adjclosedf = pd.DataFrame()
adjclosedf['ADBE'] = stocks_prices['ADBE']
print(adjclosedf)

#lstm model, splitting data in sets
xtrain, ytrain, xtest, ytest, scale = xtrain_ytrain(adj_close_df=adjclosedf)
model = lstm_model(xtrain=xtrain, ytrain=ytrain)

#plot results
visualizing(model=model, xtest=xtest, ytest=ytest, sc=scale)