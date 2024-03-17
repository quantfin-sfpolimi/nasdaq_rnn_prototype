#import libraries
from helpers import xtrain_ytrain, visualizing

#load data, clean data frame (closing stock prices)



#hash data, split in sets

#lstm model
xtrain, ytrain, xtest, ytest, scale = xtrain_ytrain(adj_close_df=stock_prices)
model = lstm_model(xtrain=xtrain, ytrain=ytrain)
#plot results
visualizing(model=model, xtest=xtest, ytest=ytest, sc=scale)