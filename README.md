# Stock Price Prediction with LSTM

This repository contains code for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The code is structured into several files for better organization and readability.

## Code Structure

prototype.py: The main script that imports helper functions and executes the stock price prediction process.
helpers.py: Contains various helper functions for data loading, preprocessing, LSTM model building, visualization, and model evaluation.
requirements.txt: A file listing all the required Python libraries for running the code.
README.md: This file, providing an overview of the project and instructions.
Usage
To run the stock price prediction code:

1. Install the required Python libraries listed in `requirements.txt`.
2. Execute the `main.py` script.

## How It Works

Data Loading: The code loads historical stock price data from Yahoo Finance using the yfinance library.
Data Preprocessing: It cleans the dataset by removing any NaN values and normalizing the data.
Model Training: An LSTM model is trained using the preprocessed data.
Prediction and Visualization: The trained model is used to predict future stock prices, and the results are visualized using matplotlib.
Model Evaluation: The model's performance is evaluated based on the loss over epochs during training.

## Streamlit App

Check out the [App](https://nasdaqrnn-prototype-sfqf.streamlit.app/)
built with Streamlit for an interactive demonstration of the prediction process.

### Contributors
Giulia Talà - @giutala
Alberto Toia - @AlbertoToia
Simone Zani - @zanisimone

Note: This project is for educational purposes (Starting Finance Polimi, Quantitative Finance Division, Politecnico di Milano) only and should not be used for actual stock trading decisions.
