"""
preprocess.py
-------------
Handles all data loading, cleaning, and feature engineering
for the Customer Churn Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps:
    - Fix TotalCharges dtype (comes as string with spaces)
    - Drop customerID (not a feature)
    - Remove rows with missing values
    """
    df = df.copy()

    # TotalCharges has whitespace strings — coerce to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop customer ID — not a predictive feature
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing values")

    print(f"[INFO] Clean data shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - Create tenure_group (new vs mid vs long-term customer)
    - Create avg_monthly_spend
    """
    df = df.copy()

    # Tenure buckets
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 36, 72],
        labels=['0-1 yr', '1-3 yrs', '3+ yrs'],
        include_lowest=True
    )

    # Charge ratio: monthly vs total (signals loyalty)
    df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Encode categorical variables and scale numeric features.
    Returns: X (features), y (target), feature_names
    """
    df = df.copy()

    # Separate target
    y = (df['Churn'] == 'Yes').astype(int)
    df.drop(columns=['Churn'], inplace=True)

    # One-hot encode all remaining object/category columns
    df = pd.get_dummies(df, drop_first=True)

    # Scale numeric features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charge_ratio']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"[INFO] Final feature matrix: {df.shape}")
    return df, y, df.columns.tolist(), scaler


def run_preprocessing(filepath: str):
    """Full pipeline: load → clean → engineer → encode."""
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    X, y, feature_names, scaler = encode_and_scale(df)
    return X, y, feature_names, scaler


if __name__ == "__main__":
    X, y, features, scaler = run_preprocessing("../data/churn.csv")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Churn rate in dataset: {y.mean():.1%}")
