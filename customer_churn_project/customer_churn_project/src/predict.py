"""
predict.py
----------
Load the saved model and make churn predictions
on new customer data.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_data, engineer_features


MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')


def load_artifacts():
    model         = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
    scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    return model, scaler, feature_names


def predict(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of raw customer records,
    return churn predictions with probabilities.
    """
    model, scaler, feature_names = load_artifacts()

    df = clean_data(input_df)
    df = engineer_features(df)

    # Drop target if accidentally included
    df.drop(columns=['Churn'], inplace=True, errors='ignore')

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Align columns to training features
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charge_ratio']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    predictions  = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = input_df.copy()
    results['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
    results['Churn_Probability'] = np.round(probabilities, 4)
    results['Risk_Level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    return results[['customerID', 'Churn_Prediction', 'Churn_Probability', 'Risk_Level']]


if __name__ == "__main__":
    # Demo: predict on 5 random customers from the dataset
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'churn.csv')
    sample = pd.read_csv(DATA_PATH).sample(5, random_state=7)
    results = predict(sample)
    print("\n── Sample Predictions ──────────────────────────")
    print(results.to_string(index=False))
