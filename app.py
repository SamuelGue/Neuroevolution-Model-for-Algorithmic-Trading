from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

from dataclasses import dataclass

import kagglehub

from datetime import datetime
import plotly.graph_objects as go
from Model_Functionality import TargetExposureNEATModel, setup_target_exposure_neat_model, backtest_target_exposure_genome
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
    neat_model = TargetExposureNEATModel(model_config_path=config_path,prices=prices)

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
        live_slot = st.empty()

        def show_live_network(generation, best_genome, image):
            live_slot.image(image, caption=f"Generation {generation} - best fitness {best_genome.fitness:.4f}")

        winner = setup_target_exposure_neat_model(
            "config-feedforward-target-exposure.ini",
            st.session_state.training_data,
            on_generation=show_live_network,
        )
        st.session_state.winner = winner
        st.session_state.model_trained = True
        st.sidebar.success("Model trained successfully.")

    if st.session_state.model_trained:
        st.image("winner_net.png", caption="Winning Neural Network Architecture", width="stretch")

if st.session_state.model_trained:
    st.sidebar.header("Test Model")
    test_ticker = st.sidebar.selectbox("Stock ticker to test on:", symbols, key="test_ticker")
    test_balance = st.sidebar.number_input("Starting balance:", min_value=1.0, value=1000.0, key="test_balance")

    if st.sidebar.button("Run Test"):
        test_df = setup_dataframe(pd.read_csv(path+"/stocks/"+test_ticker+".csv"), period=14)
        st.session_state.test_result = (test_ticker,) + backtest_target_exposure_genome(
            st.session_state.winner,
            "config-feedforward-target-exposure.ini",
            test_df,
            balance=test_balance,
        )

    if "test_result" in st.session_state:
        result_ticker, result_df, summary = st.session_state.test_result
        st.header(f"Model Test on {result_ticker}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model return", f"{summary['model_return']:.2%}")
        col2.metric("Buy and hold return", f"{summary['buy_and_hold_return']:.2%}")
        col3.metric("Max drawdown", f"{summary['max_drawdown']:.2%}")

        fig = go.Figure(data=[go.Scatter(x=result_df["Date"], y=result_df[c], mode="lines", name=c) for c in ["Model", "Buy and Hold"]])
        fig.update_layout(title=f"{result_ticker}: Model vs Buy and Hold", yaxis=dict(title=dict(text="Equity (USD)")), xaxis=dict(title=dict(text="Date")))
        st.plotly_chart(fig, width="stretch")

        exposure_fig = go.Figure(data=[go.Scatter(x=result_df["Date"], y=result_df["Exposure"], mode="lines", name="Exposure")])
        exposure_fig.update_layout(title="Model exposure over time", yaxis=dict(title=dict(text="Exposure (fraction of equity)")), xaxis=dict(title=dict(text="Date")))
        st.plotly_chart(exposure_fig, width="stretch")