import pickle

with open("stocks_prices_dataframe.pkl", "rb") as f:
    stocks_prices = pickle.load(f)

print(stocks_prices)