#!/usr/bin/env python3
"""
EnsembleWeather: train an ensemble regressor on multiple weather forecast sources.
"""
import os
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def load_data(data_dir, target):
    # Load actuals
    actuals = pd.read_csv(os.path.join(data_dir, 'historical_actuals.csv'), parse_dates=['date'])
    actuals = actuals[['date', target]].rename(columns={target: 'actual'})
    # Load forecasts
    dfs = []
    for fname in os.listdir(data_dir):
        if fname.startswith('historical_forecast_') and fname.endswith('.csv'):
            src = fname.replace('historical_forecast_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, fname), parse_dates=['date'])
            df = df[['date', target]].rename(columns={target: src})
            dfs.append(df)
    # Merge all
    data = actuals
    for df in dfs:
        data = data.merge(df, on='date', how='inner')
    data = data.dropna()
    return data

def train_ensemble(data):
    X = data.drop(columns=['date', 'actual'])
    y = data['actual']
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    print(f"Training MSE: {mse:.4f}")
    weights = pd.Series(model.coef_, index=X.columns)
    print("Learned weights:")
    print(weights)
    return model, weights

def main():
    parser = argparse.ArgumentParser(description='Train ensemble weather model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with CSV data files')
    parser.add_argument('--target', type=str, required=True, help='Target variable name')
    parser.add_argument('--output', type=str, default='ensemble_model.joblib', help='Output model file')
    args = parser.parse_args()

    data = load_data(args.data_dir, args.target)
    model, weights = train_ensemble(data)
    joblib.dump({'model': model, 'weights': weights}, args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()
