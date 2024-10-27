# Time Series Forecasting (BTC RNN Project)

## Supplemental Blog Post

For a detailed explanation of the project, including the preprocessing method, model architecture, results, and personal insights, please refer to my blog post:<br>
[Predicting Future Bitcoin Close Price Using Time Series Forecasting](https://medium.com/@masonthecount/predicting-future-bitcoin-close-price-using-time-series-forecasting-58ece48b1f08).

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)
- [Supplemental Blog Post](#supplemental-blog-post)

## Project Overview

This project aims to forecast Bitcoin (BTC) prices using recurrent neural networks (RNNs). By leveraging historical BTC data from Coinbase and Bitstamp, the model predicts the closing price of BTC for the next hour based on the past 24 hours of data. The project involves data preprocessing, model training, and validation.

## Key Features

- **RNN Architecture**: Utilizes a recurrent neural network to capture temporal patterns in BTC price data.
- **Data Preprocessing**: Implements a comprehensive data preprocessing script to ensure clean and relevant input for the model.
- **Mean Squared Error Loss**: Employs mean squared error as the cost function for model training.
- **TF Data Pipeline**: Uses TensorFlow's tf.data.Dataset for efficient data feeding during model training.

## Prerequisites

- Python
- TensorFlow
- NumPy
- Pandas

## Task Summaries

- **preprocess_data.py**: Preprocess the raw BTC data, considering factors such as feature relevance, data scaling, and saving the processed data for training.

- **forecast_btc.py**: Create, train, and validate a Keras model for forecasting BTC prices using past 24 hours of data.
