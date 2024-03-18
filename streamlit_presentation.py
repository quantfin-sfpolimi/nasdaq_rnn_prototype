import streamlit as st
import pandas as pd
import pickle
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

with open("stocks_prices_dataframe.pkl", "rb") as f:
    stocks_prices = pickle.load(f)
    
st.title('NASDAQ RNN Project, the prototype')
st.write('The goal of the work done during this week, where we worked to deliver a prototype of what will, in three months, be the final project, was to throw ourselves head-first towards the actual programming that the project entailed.')
st.write('The first step was to preprocess the data, which you can see in the following graph:')
st.line_chart(stocks_prices.iloc[:, :10], width=1)
st.write('But what do we do with the data? I\'m glad you asked :)')
st.write('Apart from drawing pretty graphs, we also worked with the data!')
pr = stocks_prices.profile_report()
st_profile_report(pr)