import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt

# Load the stock data
data = pd.read_csv('AAPL_raw.csv', index_col='Date', parse_dates=True)

# Select the features you want to use for training
features = ['Open', 'High', 'Low', 'Close', 'EMA4', 'EMA8', 'EMA16']
data = data[features]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data[features] = scaler.fit_transform(data[features])

# Create the training and testing datasets
train_data = data[:-300]
test_data = data[-300:]

# Create the training and testing labels
train_labels = train_data['Close'].shift(-1).dropna()
test_labels = test_data['Close'].shift(-1).dropna()

# Create the training and testing sequences
def create_sequences(data, labels, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = labels[i + seq_length]
        X.append(_x)
        y.append(_y)
    return np.array(X), np.array(y)

seq_length = 10
train_X, train_y = create_sequences(train_data, train_labels, seq_length)
test_X, test_y = create_sequences(test_data, test_labels, seq_length)

# Define the CNN-DNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, len(features))))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Increased number of units
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))  # Added another dense layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')  # Reduced learning rate

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y))  # Increased epochs

# Evaluate the model
loss = model.evaluate(test_X, test_y, verbose=0)
print('Test Loss:', loss)

# Make predictions
predictions = model.predict(test_X)

# Inverse transform the predictions to get the original scale
# Use the 'Close' feature scaler to inverse transform the predictions
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data[['Close']])
predictions = close_scaler.inverse_transform(predictions)
test_y = close_scaler.inverse_transform(test_y.reshape(-1, 1))

plt.plot(test_y, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.savefig('stock_price_prediction.png')
plt.ion()
plt.pause(100)