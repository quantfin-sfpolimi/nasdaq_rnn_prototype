import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
import streamlit as st

st.title('NASDAQ RNN Project, the prototype')




df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
pr = df.profile_report()

st_profile_report(pr)