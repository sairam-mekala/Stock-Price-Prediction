# Stock Price Prediction using LSTM and Neptune

## Overview
This project utilizes an LSTM (Long Short-Term Memory) neural network to predict stock prices based on historical data. It employs TensorFlow/Keras for deep learning, Neptune for experiment tracking, and Matplotlib for visualization.

## Features
- Loads stock price data from a CSV file
- Preprocesses and scales the data using MinMaxScaler
- Uses a sliding window approach to create training data
- Implements an LSTM model for time series prediction
- Tracks the training process and logs results using Neptune
- Evaluates model performance using RMSE (Root Mean Squared Error)
- Plots and uploads prediction results to Neptune

## Requirements
Ensure you have the following installed:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Neptune

Install dependencies using:
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib neptune
```

## Setup
1. Replace `your project name` and `your api token` in the `neptune.init_run()` function with your Neptune project details.
2. Update the dataset path in `pd.read_csv('C:/Users/mekal/Downloads/AAPL (2).csv')`.

## Running the Script
Run the script using:
```bash
python your_script.py
```

## Neptune Tracking
- RMSE is logged under `lstm/rmse`.
- A plot of actual vs predicted prices is uploaded to Neptune.

## Results
The script outputs:
- Predicted stock prices plotted against actual prices
- RMSE value to measure prediction accuracy

## Notes
- Ensure your dataset has a 'Date' column and an 'Adj Close' column.
- The model trains for 20 epochs; modify this as needed.

## License
This project is open-source. Feel free to modify and enhance it.


![Plot of Stock Trends](https://github.com/user-attachments/assets/25031f26-7394-4267-8d13-62f148a361d7)
