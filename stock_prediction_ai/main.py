from features.feature_engineering import extract_features
from features.vae_denoising import denoise_features
from gan.train_gan import train_gan
from optimization.bayesian_optimization import optimize_hyperparams
from drl.train_drl import train_drl_agent


def main():
    # 1. Data Preprocessing & Feature Engineering
    features = extract_features('data/stock_data.csv')
    high_level_features = denoise_features(features)

    # 2. GAN Training
    best_gan_params = optimize_hyperparams(high_level_features)
    gan_model = train_gan(high_level_features, best_gan_params)

    # 3. DRL Training
    train_drl_agent(gan_model)


if __name__ == "__main__":
    main()