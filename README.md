README.md

EnsembleWeather

This project builds an ensemble machine learning model to forecast weather by training on historical data from multiple online weather services. By learning optimal weights for each source, the ensemble aims to outperform individual models.

Requirements

Python 3.8+

pandas

scikit-learn

joblib

Installation

pip install pandas scikit-learn joblib

Data

Place CSV files in data/ with the following format per file (one file per source):

date,temperature,humidity,wind_speed,...
date,actual_temperature

historical_forecast_<source>.csv: forecasts from each service

historical_actuals.csv: actual observed values

Usage

python ensemble_weather.py --data-dir data/ --target temperature

This script reads forecasts and actuals, trains a weighted linear regression to combine sources, and saves the model weights.

Customization

Change --target to other variables (e.g., humidity).

Replace the regression model in ensemble_weather.py with another estimator (e.g., Ridge, RandomForestRegressor).
