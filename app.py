from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
import json

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable

import kagglehub
import websockets
import sys
import requests
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import plotly.graph_objects as go
from Evaluators import TopicEvaluator
from Model_Functionality import NEATmodel, setup_neat_model
from ModelInputs import setup_dataframe
import time

indicator_descriptions = json.loads(open("indicator_descriptions.json").read())

st.title("ANUSHKA - Algorithmic Trading Platform")
st.header("A platform for algorithmic trading and financial analysis.")
st.subheader("See the code on [Github](https://github.com/SamuelGue/Neuroevolution-Model-for-Algorithmic-Trading)")

st.write("This platform allows users to train and evaluate trading strategies using neuroevolution techniques. It provides a user-friendly interface for configuring parameters, visualizing results, and analyzing performance metrics.")

path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")

symbols = os.listdir(path+"\\stocks")
symbols = [i.split(".")[0] for i in symbols]

st.set_page_config(
    page_title="ANUSHKA"
)


st.sidebar.header("Configuration")
tickers = st.sidebar.selectbox("Select a stock ticker:", symbols)

time_window = st.sidebar.slider("Time Window (days):", min_value=1, max_value=365, value=30)
df = pd.read_csv(path+"/stocks/"+tickers+".csv")
df = setup_dataframe(df, period=14)

df = df.tail(time_window)

fig = go.Figure(data=[go.Candlestick(x=df['Date'], 
                open=df['Open'], 
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'])])

fig.update_layout(title=f"{tickers} Stock Price", yaxis=dict(title=dict(text=f"{tickers} Stock (USD)")), xaxis=dict(title=dict(text="Date")))

st.plotly_chart(fig, width="stretch")

st.sidebar.header("Indicators")

indicators = st.sidebar.selectbox("Select Indicators", ["ADX", "CCI", "VI", "RSI", "Williams %R", "FI", "OBV", "VPT", "ADI"])

df = df.tail(time_window)
st.header("Plot of "+indicators+" Indicator")
st.text("Here you can see plots of select indicators for the selected stock ticker. Choose an indicator from the sidebar to visualize it over the selected time window. Read below the diagram for more information about how to interpret the information displayed here.")
st.text(f"Interpretation: {indicator_descriptions[indicators]['interpretation']}")
fig = go.Figure(data=[go.Scatter(x=df['Date'], y=df[i], mode='lines', name=i) for i in indicator_descriptions[indicators]["columns"]])
for i in indicator_descriptions[indicators]["y_lines"]:
    fig.add_hline(y=i, annotation_text=str(i), line_color="green", annotation_position="bottom right")

st.plotly_chart(fig, width="stretch")

st.sidebar.header("Training Configuration")

training_ticker_options = st.sidebar.multiselect(
    "Stock Tickers for Training",
    symbols
)

def initialise_training_data(selected_tickers):
    training_data = []
    for ticker in selected_tickers:
        temp_df = pd.read_csv(path+"/stocks/"+ticker+".csv")
        df = setup_dataframe(temp_df, period=14)
        training_data.append(df)
    return training_data

def train_model(training_data, config_path):
    prices = training_data
    print(prices)
    time.sleep(5)
    neat_model = NEATmodel(model_config_path=config_path,prices=prices)

    winner = neat_model.train()
    neat_model._create_graph(winner)
    st.sidebar.success("Model trained successfully.")

    return winner

if "training_data_ready" not in st.session_state:
    st.session_state.training_data_ready = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if st.sidebar.button("Confirm"):
    st.session_state.training_data = initialise_training_data(training_ticker_options)
    st.session_state.training_data_ready = True
    st.sidebar.success("Training data initialized successfully.")

if st.session_state.training_data_ready:
    if st.sidebar.button("Train Model"):
        train_model(st.session_state.training_data, "config-feedforward.ini")
        st.session_state.model_trained = True
        st.sidebar.success("Model trained successfully.")

    if st.session_state.model_trained:
        st.image("winner-net.png", caption="Winning Neural Network Architecture", width="stretch")