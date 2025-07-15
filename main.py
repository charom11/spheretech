from features.feature_engineering import extract_features
from features.vae_denoising import denoise_features
from gan.train_gan import train_gan
from optimization.bayesian_optimization import optimize_hyperparams
from drl.train_drl import train_drl_agent
from models.lstm_model import LSTMModel
from utils.fetch_data import fetch_and_save_stock_data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pandas as pd
import time
import sys


def prepare_lstm_data(df, feature_cols, target_col, seq_len=10):
    data = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(targets[i+seq_len])
    return np.array(X), np.array(y)


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            model.eval()
            val_output = model(X_val)
            val_loss = criterion(val_output.squeeze(), y_val)
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    return model


def predict_future(model, last_seq, n_steps, scaler_y):
    model.eval()
    preds = []
    seq = last_seq.copy()
    for _ in range(n_steps):
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # ensure float32
        with torch.no_grad():
            pred = model(inp).item()
        preds.append(pred)
        # Append prediction to sequence and remove first element
        seq = np.vstack([seq[1:], np.append(seq[-1][:-1], pred)])
    # Inverse transform predictions
    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


def main():
    # Accept CSV file path as argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        asset = 'BTC-USD'  # or infer from filename if needed
        plot_path = os.path.splitext(data_file)[0] + '_lstm_pred_vs_actual.png'
    else:
        asset = 'BTC-USD'
        data_file = 'stock_prediction_ai/data/Bitcoin_1_1_2008-7_15_2025_historical_data_coinmarketcap.csv'
        plot_path = 'stock_prediction_ai/data/lstm_pred_vs_actual_btc_2021_2025.png'
    if not os.path.exists(data_file):
        fetch_and_save_stock_data(asset, '2024-01-01', '2024-07-01', data_file)
    features = extract_features(data_file)
    print(f'Extracted Features for {asset}:')
    print(features.head())
    high_level_features = denoise_features(features)
    print('High-Level Features (after denoising):')
    print(high_level_features.head())

    # Use new features
    feature_cols = ['Close', 'MA_5', 'MA_10', 'RSI_14', 'MACD', 'MACD_signal', 'EMA_20', 'EMA_50', 'BB_MID', 'BB_UPPER', 'BB_LOWER', 'Volume']
    target_col = 'Close'
    high_level_features = high_level_features.dropna()

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_raw = high_level_features[feature_cols].values.astype(np.float32)
    y_raw = high_level_features[[target_col]].values.astype(np.float32)
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)
    # Prepare LSTM data
    seq_len = 10
    X, y = [], []
    for i in range(len(X_scaled) - seq_len):
        X.append(X_scaled[i:i+seq_len])
        y.append(y_scaled[i+seq_len])
    X = np.array(X)
    y = np.array(y).flatten()
    print(f'LSTM Input Shape: {X.shape}, Target Shape: {y.shape}')
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    split = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]
    # Use deeper, bidirectional LSTM
    input_size = X.shape[2]
    hidden_size = 128
    num_layers = 3
    output_size = 1
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional=True)
    # Early stopping logic
    start_time = time.time()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    max_minutes = 5
    epochs = 1000
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        output = lstm_model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            lstm_model.eval()
            val_output = lstm_model(X_val)
            val_loss = criterion(val_output.squeeze(), y_val)
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if (time.time() - start_time) > max_minutes * 60:
            print(f"Stopped training after {max_minutes} minutes at epoch {epoch}")
            break
    with torch.no_grad():
        preds_scaled = lstm_model(X_val).squeeze().numpy()
    # Inverse transform predictions and actuals
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    print('LSTM Predictions (first 5):')
    print(preds[:5])
    # Visualize predictions vs. actual
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(y_val_orig, label='Actual')
    plt.plot(preds, label='Predicted')
    plt.title(f'LSTM Predictions vs. Actual ({asset})')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')
    plt.close()
    # Predict next 30 days
    n_future = 30
    last_seq = X_scaled[-seq_len:]
    future_preds = predict_future(lstm_model, last_seq, n_future, scaler_y)
    print(f'Next {n_future} predicted prices for BTC:')
    print(future_preds)
    # Save future predictions with dates
    last_date_str = high_level_features.iloc[-1, 0]
    try:
        last_date = datetime.strptime(str(last_date_str), '%Y-%m-%d')
    except Exception:
        last_date = pd.to_datetime(last_date_str)
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(n_future)]
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds})
    future_df.to_csv(os.path.splitext(data_file)[0] + '_future_pred.csv', index=False)
    # Print actual vs. predicted for future (if actuals available)
    if len(high_level_features) >= n_future:
        actual_future = high_level_features[target_col].values[-n_future:]
        print('Date       | Actual      | Predicted')
        for d, a, p in zip(future_dates, actual_future, future_preds):
            print(f'{d} | {a:.2f} | {p:.2f}')

    # 3. GAN Training (placeholder)
    best_gan_params = optimize_hyperparams(high_level_features)
    print('Best GAN Params:')
    print(best_gan_params)
    gan_model = train_gan(high_level_features, best_gan_params)
    print('Trained GAN Model:')
    print(gan_model)
    print('Training DRL Agent...')
    train_drl_agent(gan_model)
    print('DRL Agent Training Complete.')

if __name__ == "__main__":
    main()