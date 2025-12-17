from fastapi import FastAPI
import pandas as pd
import mlflow
from datetime import datetime
from api.schemas import PredictionInput, PredictionOutput

# IMPORTANT: must already be set
mlflow.set_tracking_uri(None)

MODEL_NAME = "GBM"

app = FastAPI(title="Bike Sharing Prediction API")

def prepare_features(df: pd.DataFrame):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    return df.drop(columns=["datetime"])

def load_model(version: str):
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{version}"
    )

@app.post("/predict_gbm", response_model=PredictionOutput)
def predict_gbm(data: PredictionInput):
    model = load_model("1")
    df = pd.DataFrame([data.dict()])
    X = prepare_features(df)
    pred = model.predict(X)[0]

    return {
        "model_name": "GBM",
        "model_version": "1",
        "prediction": float(pred)
    }

@app.post("/predict_xgboost", response_model=PredictionOutput)
def predict_xgboost(data: PredictionInput):
    model = load_model("2")
    df = pd.DataFrame([data.dict()])
    X = prepare_features(df)
    pred = model.predict(X)[0]

    return {
        "model_name": "XGBoost",
        "model_version": "2",
        "prediction": float(pred)
    }

@app.post("/predict_random_forest", response_model=PredictionOutput)
def predict_rf(data: PredictionInput):
    model = load_model("3")
    df = pd.DataFrame([data.dict()])
    X = prepare_features(df)
    pred = model.predict(X)[0]

    return {
        "model_name": "RandomForest",
        "model_version": "3",
        "prediction": float(pred)
    }
