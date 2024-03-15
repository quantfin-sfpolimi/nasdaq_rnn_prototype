#import libraries
from helpers import xtrain_ytrain, visualizing, load_dataframe

#load data, clean data frame (closing stock prices)
stock_prices = load_dataframe()



#hash data, split in sets

#lstm model
## for albi:: passo alla xtrain_ytrain il return dal pkl trasformato in df (l'ho chiamato stock_prices),
## all'interno di questa funzione c'e' già il hashing, che mi ritorna train e test
xtrain, ytrain, testset = xtrain_ytrain(adj_close_df=stock_prices)

#plot results
visualizing(xtrain=xtrain, ytrain=ytrain, xtest=testset)