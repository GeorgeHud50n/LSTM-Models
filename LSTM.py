## Download Data
import yfinance as yf
import pandas as pd

def download_data(currency_pair):
    data = yf.download(currency_pair, start="2015-01-01", end="2023-07-07")
    return data

data = download_data('EURUSD=X')

## Data Preprocess 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Keep only the close price
data = data[['Close']]

# Split into train and test set
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]
test_index = test.index

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)

# Create training dataset
look_back = 14
trainX, trainY = [], []

for i in range(look_back, len(train)):
    trainX.append(scaled_data[i-look_back:i, 0])
    trainY.append(scaled_data[i, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

## Build LSTM 
import tensorflow as tf

# Model 1
model1 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Model 2
model2 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=False, input_shape=(trainX.shape[1], 1)),
    tf.keras.layers.Dense(1)
])

# Model 3
model3 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Model 4
model4 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(trainX.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)
])

models = [model1, model2, model3, model4]
predictions = []

for i, model in enumerate(models):
    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, batch_size=1, epochs=1)

    ## Make Predictions 

    # Create the testing dataset
    test_data = scaler.transform(test)

    testX = []
    for i in range(look_back, len(test_data)):
        testX.append(test_data[i-look_back:i, 0])

    testX = np.array(testX)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # Get the models predicted price values 
    prediction = model.predict(testX)
    prediction = scaler.inverse_transform(prediction)

    predictions.append(prediction)

    # Print RMSE of model
    rmse = np.sqrt(mean_squared_error(test[-testX.shape[0]:], prediction))
    print(f'RMSE of LSTM Model {i+1}: {rmse}')

## Compare 

import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
plt.plot(test, label='Actual')
for i, prediction in enumerate(predictions):
    plt.plot(pd.DataFrame(prediction, index=test.index[-testX.shape[0]:], columns=['Close']), label=f'LSTM Model {i+1}')
plt.legend()
plt.show()
