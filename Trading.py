import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

# Initialize Neptune
run = neptune.init_run(
    project="your project name",
    api_token="your api tocken"
)

# Load your own dataset
data = pd.read_csv('C:/Users/mekal/Downloads/AAPL (2).csv')

# Ensure the dataset has a 'Date' and 'Adj Close' column
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Adj Close']]

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

# Prepare data
window_size = 60
X = []
y = []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM
lstm_model.fit(X_train_lstm, y_train, batch_size=32, epochs=20, verbose=1)

# Make predictions
lstm_pred = lstm_model.predict(X_test_lstm)
lstm_pred = scaler.inverse_transform(lstm_pred)  # Inverse scaling
y_test_inv = scaler.inverse_transform(y_test)  # Inverse scaling

# Calculate RMSE
lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred))

# Log to Neptune
run["lstm/rmse"] = lstm_rmse

# Plot predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index[train_size + window_size:], y_test_inv, color='blue', label='Actual Prices')
plt.plot(data.index[train_size + window_size:], lstm_pred, color='red', label='LSTM Predictions')
plt.title('Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
run['Plot of Stock Trends'].upload(neptune.types.File.as_image(plt.gcf()))
plt.show()

# Stop Neptune
run.stop()
