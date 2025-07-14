from features.feature_engineering import extract_features
from features.vae_denoising import denoise_features
from gan.train_gan import train_gan
from optimization.bayesian_optimization import optimize_hyperparams
from drl.train_drl import train_drl_agent
from models.lstm_model import LSTMModel
import torch
import numpy as np


def prepare_lstm_data(df, feature_cols, target_col, seq_len=10):
    data = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(targets[i+seq_len])
    return np.array(X), np.array(y)


def main():
    # 1. Data Preprocessing & Feature Engineering
    features = extract_features('stock_prediction_ai/data/aapl_2024.csv')
    print('Extracted Features:')
    print(features.head())
    high_level_features = denoise_features(features)
    print('High-Level Features (after denoising):')
    print(high_level_features.head())

    # 2. LSTM Model Integration
    feature_cols = ['Close', 'MA_5', 'MA_10', 'RSI_14', 'MACD', 'MACD_signal']
    target_col = 'Close'
    # Drop rows with NaN (from rolling calculations)
    high_level_features = high_level_features.dropna()
    X, y = prepare_lstm_data(high_level_features, feature_cols, target_col)
    print(f'LSTM Input Shape: {X.shape}, Target Shape: {y.shape}')
    # Convert to torch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    # Instantiate LSTM model
    input_size = X.shape[2]
    hidden_size = 32
    num_layers = 2
    output_size = 1
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    # Forward pass (prediction)
    with torch.no_grad():
        preds = lstm_model(X_tensor)
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