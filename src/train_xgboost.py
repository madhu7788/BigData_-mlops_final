import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EXPERIMENT_NAME = "bike_sharing_time_series"
TARGET = "count"

def prepare_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    return df.drop(columns=["datetime"])

def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validate.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = prepare_features(train.drop(columns=[TARGET]))
    y_train = train[TARGET]

    X_val = prepare_features(val.drop(columns=[TARGET]))
    y_val = val[TARGET]

    X_test = prepare_features(test.drop(columns=[TARGET]))
    y_test = test[TARGET]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    with mlflow.start_run(run_name="XGBoost"):
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, test_pred)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("learning_rate", 0.1)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print("XGBoost → RMSE:", rmse)

if __name__ == "__main__":
    main()
