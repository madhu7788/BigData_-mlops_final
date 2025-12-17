import os
os.environ["NANNYML_DISABLE_USAGE_LOGGING"] = "1"

import pandas as pd
from nannyml.drift.univariate import UnivariateDriftCalculator

# -----------------------------
# Feature engineering (SAME as model)
# -----------------------------
def prepare_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    return df.drop(columns=["datetime"])

# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

train_fe = prepare_features(train)
test_fe = prepare_features(test)

# Select only feature columns used by model
feature_columns = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "weekday"
]

reference = train_fe[feature_columns]
analysis = test_fe[feature_columns]

# -----------------------------
# Drift calculator
# -----------------------------
drift_calculator = UnivariateDriftCalculator(
    column_names=feature_columns
)

drift_calculator.fit(reference)
drift_results = drift_calculator.calculate(analysis)

# -----------------------------
# Output
# -----------------------------
print("✅ Drift calculation completed\n")
print(drift_results.to_df().head())

# Save results for report
drift_results.to_df().to_csv(
    "reports/nannyml_drift_results.csv",
    index=False
)
