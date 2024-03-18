#import libraries
import pandas as pd
from helpers import xtrain_ytrain, visualizing, load_dataframe, clean_df, lstm_model

#load data, clean data frame (closing stock prices)
stocks_prices, tickers = load_dataframe()
clean_df(10, tickers=tickers, stocks_prices=stocks_prices)
# clean remaining NaN
stocks_prices.dropna(inplace = True)
# get one stock, only for prototype purpose (ADBE)
stock_prices = pd.DataFrame()
stock_prices['ADBE'] = stocks_prices['ADBE']
print(stock_prices)
#hash data, split in sets

#lstm model
xtrain, ytrain, xtest, ytest, scale = xtrain_ytrain(adj_close_df=stock_prices)
model = lstm_model(xtrain=xtrain, ytrain=ytrain)
#plot results
visualizing(model=model, xtest=xtest, ytest=ytest, sc=scale)