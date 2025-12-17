import pandas as pd
import mlflow.pyfunc

# 1. Load model from MLflow Registry (Champion = GBM)
MODEL_NAME = "GBM"
Version = "1"   # or "None" if you didn't set stage

model_uri = f"models:/GBM/None"
model = mlflow.pyfunc.load_model(model_uri)

print("✅ Model loaded successfully")

# 2. Sample input (ONE RECORD)
data = {
    "datetime": ["2012-06-15 09:00:00"],
    "season": [2],
    "holiday": [0],
    "workingday": [1],
    "weather": [1],
    "temp": [0.65],
    "atemp": [0.62],
    "humidity": [0.55],
    "windspeed": [0.18]
}

df = pd.DataFrame(data)

# 3. Feature engineering (same logic as main.py)
df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday
df = df.drop(columns=["datetime"])

# 4. Predict
prediction = model.predict(df)

print("🎯 Prediction output:", prediction)
