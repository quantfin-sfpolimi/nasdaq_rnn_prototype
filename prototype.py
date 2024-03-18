#import libraries
from helpers import xtrain_ytrain, visualizing, load_dataframe, clean_df, lstm_model
import yfinance as yf

#load data, clean data frame (closing stock prices)
stocks_prices, tickers = load_dataframe()
clean_df(10, tickers=tickers, stocks_prices=stocks_prices)

# clean remaining NaN

# get one stock, only for prototype purpose (ADBE)
start = '2012-01-01'
end = '2022-12-21'
stock = 'ADBE'
stocks_prices = yf.download(stock, start, end)
adjclosedf = stocks_prices['Adj Close']
adjclosedf.dropna(inplace = True)

print(adjclosedf)

#lstm model, splitting data in sets
xtrain, ytrain, xtest, ytest, scale = xtrain_ytrain(adj_close_df=adjclosedf)

model = lstm_model(xtrain=xtrain, ytrain=ytrain)
#plot results
visualizing(model=model, xtest=xtest, ytest=ytest, sc=scale)