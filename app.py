from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# --- Predictor Class ---
class SimplePredictor:
    def __init__(self, model_path, historical_csv_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.historical = self._load_historical(historical_csv_path)

    def _load_historical(self, path):
        df = pd.read_csv(path)

        # Convert settlement_date safely to datetime
        df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')
        df = df.dropna(subset=['settlement_date'])

        # Ensure settlement_period is numeric
        df['settlement_period'] = pd.to_numeric(df['settlement_period'], errors='coerce')
        df = df.dropna(subset=['settlement_period'])

        # Create combined datetime field
        df['full_datetime'] = df.apply(
            lambda row: row['settlement_date'] + timedelta(minutes=(int(row['settlement_period']) - 1) * 30),
            axis=1
        )

        return df.set_index('full_datetime')

    def create_features(self, df, n_lags=24):
        df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        df['full_datetime'] = df.apply(
            lambda row: row['settlement_date'] + timedelta(minutes=(row['settlement_period'] - 1) * 30),
            axis=1
        )

        for i in range(1, n_lags + 1):
            lag_time = df['full_datetime'] - timedelta(minutes=30 * i)
            df[f'lag_{i}'] = self.historical['england_wales_demand'].reindex(lag_time.values).values

        df['hour'] = df['full_datetime'].dt.hour
        df['dayofweek'] = df['full_datetime'].dt.dayofweek
        df['month'] = df['full_datetime'].dt.month

        df = df.drop(columns=['full_datetime'])
        df = df.fillna(0)  # basic imputation

        return df[[col for col in df.columns if col.startswith('lag_') or col in ['hour', 'dayofweek', 'month']]]

    def predict(self, input_df):
        features = self.create_features(input_df)
        dmatrix = xgb.DMatrix(features)
        return self.model.predict(dmatrix)

# --- Init Predictor ---
predictor = SimplePredictor('forecast.json', 'historical_data.csv')

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    preds = predictor.predict(df)

    results = []
    for i, row in df.iterrows():
        s_date = pd.to_datetime(row['settlement_date'])
        s_period = int(row['settlement_period'])
        pred_time = s_date + timedelta(minutes=(s_period - 1) * 30)

        results.append({
            "settlement_date": pred_time.strftime('%Y-%m-%d'),
            "settlement_period": s_period,
            "predicted_demand": float(preds[i])
        })

    return jsonify(results)

# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True)
