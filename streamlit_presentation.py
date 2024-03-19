import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

with open("./stocks_prices_dataframe.pkl", "rb") as f:
    stocks_prices = pickle.load(f)

### INTRO SECTION
st.title("SFQF - Stocks price predictor")
st.header("Built as a prototype for a major project: Trading with correlation usign RNNs")
st.caption('The goal of the work done during this week, where we worked to deliver a prototype of what will, in three months, be the final project, was to throw ourselves head-first towards the actual programming that the project entailed.')

st.divider()



### FIRST SECTION

st.subheader('1. From row data to clean dataframe')
st.write('The first step was to preprocess the row data, which you can see in the following dataset:')

code = '''def clean_df(percentage, tickers, stocks_prices):
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
    return stocks_prices'''
st.code(code, language="python")

st.caption("Data taken from yahoo finance")
st.dataframe(stocks_prices)
#st.line_chart(stocks_prices)

st.divider()


### SECOND SECTION

st.subheader('2. Plot results and choose one stock')
st.write('Now that we have clean data, we can move on.')
st.write('For the sake of this prototype, we decided to use ADBE for predicting his future price, using Machine Learning methods.')

st.line_chart(stocks_prices[['ADBE', 'META', 'AAPL', 'TSLA', 'NFLX']], width=1)
st.write('As you can see in this chart, NFLX and ADBE seem to move similary, with different magnitude. That\'s exactly what we\'re looking for.')
st.write('We can extract ADBE, NFLX & META and look just for them in a simpler table')

# Pick last n elements od ADBE price and convert to numpy, then normalize it for better representation
adbe_prices = stocks_prices['ADBE'].to_numpy()[:90]
nflx_prices = stocks_prices['NFLX'].to_numpy()[:90]
meta_prices = stocks_prices['META'].to_numpy()[:90]

df = pd.DataFrame(
    {
        "Stock": ["ADBE", "NFLX", "META"],
        "prices": [adbe_prices, nflx_prices, meta_prices],
    }
)
st.dataframe(
    df,
    column_config={
        "prices": st.column_config.LineChartColumn(
            "Price (past 90 days)", y_min=60, y_max=70
        ),
    },
    hide_index=True,
)

st.divider()

### THIRD SECTION

st.subheader('3. Creating useful functions for higher speed and quality')
st.write('We used some tips for a better and efficient code, for example hashing and pickling.')
st.write('It\'s some low level stuff, so we\'re not gonna waste time on this.')
st.write('Here\'s some code we wrote.')

st.caption('Pickling, in order to increase speed in runtime')

code = '''def pickle_dump(stocks_prices):
    with open("stocks_prices_dataframe.pkl", "wb") as f:
        pickle.dump(stocks_prices, f)

def pickle_load(filename):
    with open(filename, "rb") as f:
        stocks_prices = pickle.load(f)
    return stocks_prices'''
st.code(code, language='python')

st.caption('Hashing:, in order to increase quality in the code, useful for the trained model')

code = '''def hashing_and_splitting(adj_close_df):
    # calculate checksum for every index
    checksum = np.array([crc32(v) for v in adj_close_df.index.values])
    # fraction of data to use for testing only
    test_ratio = 0.2
    # pick 20% of indices for testing
    test_indices = checksum < test_ratio * 2**32

    return adj_close_df[~test_indices], adj_close_df[test_indices]'''
st.code(code, language='python')


### THIRD SECTION
st.subheader('4. Training and testing the model')


st.write('But what do we do with the data? I\'m glad you asked :)')
st.write('Apart from drawing pretty graphs, we also worked with the data!')