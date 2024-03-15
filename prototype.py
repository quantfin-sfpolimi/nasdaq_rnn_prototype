#import libraries
from helpers import xtrain_ytrain, visualizing, get_nasdaq_tickers, loaded_df, clean_df

#load data, clean data frame (closing stock prices)
years = 5
stocks_prices = loaded_df(years, get_nasdaq_tickers())

#get columns names list
tickers = list(stocks_prices.columns)

#clean df
stocks_prices = clean_df(10, tickers, stocks_prices)

# clean remaining NaN
stocks_prices.dropna(inplace = True)

#hash data, split in sets

#lstm model
## for albi:: passo alla xtrain_ytrain il return dal pkl trasformato in df (l'ho chiamato stock_prices),
## all'interno di questa funzione c'e' già il hashing, che mi ritorna train e test
xtrain, ytrain, testset = xtrain_ytrain(adj_close_df=stock_prices)

#plot results
visualizing(xtrain=xtrain, ytrain=ytrain, xtest=testset)