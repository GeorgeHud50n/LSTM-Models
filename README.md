# LSTM Forex Price Prediction

This repository contains code for training and comparing different LSTM models to predict forex prices. The models are trained using historical forex price data and evaluated on a test set. The goal is to determine which LSTM model performs the best in terms of accuracy and ability to capture price trends.

## Dataset

The forex price data is downloaded using the `yfinance` library and covers the period from January 1, 2015, to July 7, 2023. The dataset consists of daily closing prices for the specified currency pair (EUR/USD).

## Data Preprocessing

The dataset is preprocessed before training the LSTM models. The following steps are performed:

1. Keep only the close price column from the dataset.
2. Split the data into a training set (80% of the data) and a test set (20% of the data).
3. Scale the data using the MinMaxScaler to bring the values between 0 and 1.
4. Create the training dataset by sliding a window of size `look_back` over the scaled training data. The input features are sequences of `look_back` previous prices, and the target is the next price.
5. Reshape the input features to be in the shape [samples, time steps, features] required by the LSTM models.

## LSTM Models

Four different LSTM models are implemented and compared:

1. LSTM Model 1: This model consists of two LSTM layers followed by two dense layers.
2. LSTM Model 2: This model has a single LSTM layer followed by a dense layer.
3. LSTM Model 3: This model consists of three LSTM layers followed by two dense layers.
4. LSTM Model 4: This model is the most complex and includes a combination of Conv1D, MaxPooling1D, and LSTM layers.

For each model, the following steps are performed:

1. Compile the model using the Adam optimizer and mean squared error loss.
2. Train the model on the training data for one epoch.
3. Make predictions on the test data.
4. Inverse scale the predicted prices to obtain the actual price values.
5. Calculate the root mean squared error (RMSE) between the actual and predicted prices.

## Results

The LSTM models' predictions are compared against the actual prices using a line plot. The plot shows the actual prices and the predictions from each LSTM model. Additionally, the RMSE value for each model is printed.

Based on the RMSE values and the plot, you can assess the performance of each LSTM model and determine which model performs the best in predicting forex prices.

Please note that forex price prediction is a complex task influenced by various factors, and the performance of the LSTM models may vary depending on the specific dataset and market conditions.

## Requirements

The following libraries are required to run the code:

- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

You can install these libraries using pip:

