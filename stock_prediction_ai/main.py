from features.feature_engineering import extract_features
from features.vae_denoising import denoise_features
from gan.train_gan import train_gan
from optimization.bayesian_optimization import optimize_hyperparams
from drl.train_drl import train_drl_agent


def main():
    # 1. Data Preprocessing & Feature Engineering
    features = extract_features('stock_prediction_ai/data/sample_stock_data.csv')
    print('Extracted Features:')
    print(features)
    high_level_features = denoise_features(features)
    print('High-Level Features (after denoising):')
    print(high_level_features)

    # 2. GAN Training
    best_gan_params = optimize_hyperparams(high_level_features)
    print('Best GAN Params:')
    print(best_gan_params)
    gan_model = train_gan(high_level_features, best_gan_params)
    print('Trained GAN Model:')
    print(gan_model)

    # 3. DRL Training
    print('Training DRL Agent...')
    train_drl_agent(gan_model)
    print('DRL Agent Training Complete.')


if __name__ == "__main__":
    main()