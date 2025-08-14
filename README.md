# LSTM Volatility Prediction for QQQ and SPY

Predict rolling volatility of QQQ (Nasdaq-100 ETF) and SPY (S&P 500 ETF) using separate LSTM models with historical market data from yFinance.

## Description

This project builds two LSTM neural networks to forecast 5-day rolling volatility for QQQ and SPY stocks. Data is fetched live via yFinance, cleaned, preprocessed with features like returns, moving averages, momentum, and lags. Models are trained on sequenced, scaled data. Includes evaluation metrics (MSE, MAE, R2) and visualizations.

## Features

- Data loading: QQQ (6 years), SPY (5 years) from yFinance.
- Cleaning: Handle duplicates, missing values, invalid data (e.g., negative prices, zero volumes).
- Preprocessing: Calculate returns, volatility (5-day std * sqrt(5)), SMAs (10/20), volume ratios, momentum (5/10), high-low ratios.
- Feature Engineering: Volatility lags (1,2,3,5), MAs (5/10), std (5), return lags (1,2,3).
- Sequencing: 10-day lookback for LSTM input.
- Scaling: MinMaxScaler for features and targets.
- Models: Separate LSTMs for QQQ and SPY with dropout, Adam optimizer.
- Evaluation: MSE, MAE, R2; prediction plots.
- Libraries: NumPy, Pandas, yFinance, Scikit-learn, TensorFlow/Keras, Matplotlib.

## Installation

1. Clone repo: `git clone https://github.com/yourusername/lstm-volatility-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`

requirements.txt:
```
numpy
pandas
yfinance
scikit-learn
tensorflow
matplotlib
```

## Usage

Run `LSTM_Final.ipynb` in Jupyter Notebook or Colab.

- Loads data automatically.
- Trains models (set seeds for reproducibility).
- Outputs predictions, metrics, plots.

Example code snippet:
```python
qqq_data = load_stock_data('QQQ')
spy_data = load_stock_data2('SPY')
# ... (full pipeline in notebook)
```

## Data

- Source: yFinance API.
- Columns: Open, High, Low, Close, Volume.
- Period: QQQ (6y), SPY (5y).
- Processed: 1490 rows (QQQ), 1237 rows (SPY) after cleaning.

## Model Architecture

For each stock:
- LSTM layers: 2 (50 units each) with Dropout (0.2).
- Dense output layer.
- Optimizer: Adam (lr=0.001).
- Loss: MSE.
- Epochs: 50, Batch: 32.
- Input shape: (10 timesteps, 16 features).

## Results

- Train/Test split: 80/20.
- Shapes: QQQ (1184 train, 296 test); SPY (981 train, 246 test).
- Metrics: Computed post-training (MSE, MAE, R2).
- Visuals: Actual vs Predicted volatility plots.

## Contributing

Fork, create branch, PR. Issues welcome.

