#!/usr/bin/env python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def create_sequences(data, look_back=60):
    rows = len(data) - look_back - 1
    X = np.zeros((rows, look_back, 1))
    Y = np.zeros((rows, 1))
    for i in range(rows):
        X[i] = data[i : (i + look_back), 0].reshape(-1, 1)
        Y[i] = data[i + look_back, 0]
    return X, Y


data_cb = merged_data["Close_cb"].values.reshape(-1, 1)
data_bs = merged_data["Close_bs"].values.reshape(-1, 1)
combined_data = np.mean(np.array([data_cb, data_bs]), axis=0)

X, Y = create_sequences(combined_data, look_back=60)
X = np.reshape(
    X, (X.shape[0], X.shape[1], 1)
)  # Reshape for LSTM [samples, time steps, features]

# splitting data so that the first 80% of the time-series is used for training
#   and we will attempt to predict the last 20% of the time-series
train_size = int(len(X) * 0.8)
trainX, testX = X[0:train_size], X[train_size:]
trainY, testY = Y[0:train_size], Y[train_size:]


# LSTM model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.LSTM(50, return_sequences=True, 
                             input_shape=(time_steps, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(train_data, epochs=100, validation_data=test_data)

test_loss = model.evaluate(test_data)

test_start_index = len(merged_data) - len(testX)
test_timestamps = merged_data["Timestamp"].iloc[
    test_start_index : test_start_index + len(testX)
]

# convert timestamps to a format suitable for plotting
test_dates = pd.to_datetime(test_timestamps, unit="s")

# plotting with timestamps
plt.figure(figsize=(22, 11))
plt.plot(test_dates, testY_inverted, label="Actual", linewidth=1)
plt.plot(test_dates, testPredict, label="Predicted", linewidth=1)
plt.title("BTC Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
