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


def prepare_lstm_data(df, feature_cols, target_col, seq_len=10):
    data = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(targets[i+seq_len])
    return np.array(X), np.array(y)


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=30, lr=0.001):
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


def main():
    # Select asset: 'AAPL', 'BTC-USD', or 'ETH-USD'
    asset = 'BTC-USD'  # Change to 'AAPL' or 'ETH-USD' as needed
    data_file = f'stock_prediction_ai/data/{asset.lower().replace("-usd", "")}_2024.csv'
    # Fetch data if not present
    if not os.path.exists(data_file):
        fetch_and_save_stock_data(asset, '2024-01-01', '2024-07-01', data_file)
    # 1. Data Preprocessing & Feature Engineering
    features = extract_features(data_file)
    print(f'Extracted Features for {asset}:')
    print(features.head())
    high_level_features = denoise_features(features)
    print('High-Level Features (after denoising):')
    print(high_level_features.head())

    # 2. LSTM Model Integration
    feature_cols = ['Close', 'MA_5', 'MA_10', 'RSI_14', 'MACD', 'MACD_signal']
    target_col = 'Close'
    high_level_features = high_level_features.dropna()
    X, y = prepare_lstm_data(high_level_features, feature_cols, target_col)
    print(f'LSTM Input Shape: {X.shape}, Target Shape: {y.shape}')
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    # Split train/val
    split = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]
    input_size = X.shape[2]
    hidden_size = 32
    num_layers = 2
    output_size = 1
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    # Train LSTM
    lstm_model = train_lstm(lstm_model, X_train, y_train, X_val, y_val, epochs=30)
    # Forward pass (prediction)
    with torch.no_grad():
        preds = lstm_model(X_val)
    print('LSTM Predictions (first 5):')
    print(preds[:5].squeeze().numpy())

    # 3. GAN Training (placeholder)
    best_gan_params = optimize_hyperparams(high_level_features)
    print('Best GAN Params:')
    print(best_gan_params)
    gan_model = train_gan(high_level_features, best_gan_params)
    print('Trained GAN Model:')
    print(gan_model)

    # 4. DRL Training (placeholder)
    print('Training DRL Agent...')
    train_drl_agent(gan_model)
    print('DRL Agent Training Complete.')


if __name__ == "__main__":
    main()