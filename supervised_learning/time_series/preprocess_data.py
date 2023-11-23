#!/usr/bin/env  python3
"""
preprocess data for use by forecast_btc.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocessData():
    """
    preprocess data for use by forecast_btc.py
    """
    # paths to the datasets
    coinbase_path = "data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    bitstamp_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

    # Load the datasets
    cb = pd.read_csv(coinbase_path)
    bs = pd.read_csv(bitstamp_path)

    # convert timestamps, indicating index values as seconds in the data series
    cb["Timestamp"] = pd.to_datetime(cb["Timestamp"], unit="s")
    bs["Timestamp"] = pd.to_datetime(bs["Timestamp"], unit="s")

    # fill N/A values
    cb.fillna(method="ffill", inplace=True)
    bs.fillna(method="ffill", inplace=True)

    # feature engineering and selection
    # close chosen as singular parameter/time-series datapoint to focus on
    cb = cb[["Timestamp", "Close"]]
    bs = bs[["Timestamp", "Close"]]

    # Data Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    cb.loc[:, "Close"] = scaler.fit_transform(cb[["Close"]])
    bs.loc[:, "Close"] = scaler.fit_transform(bs[["Close"]])

    # merge the datasets according to timestamp
    merged_data = pd.merge(cb, bs, on="Timestamp", suffixes=("_cb", "_bs"))

    # to subsample the data (ex. only use every 60 minutes of data)
    merged_data = merged_data.iloc[::60, :]
