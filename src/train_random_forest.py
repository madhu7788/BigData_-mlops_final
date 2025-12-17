import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EXPERIMENT_NAME = "bike_sharing_time_series"
TARGET = "count"


def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def prepare_features(df):
    df = df.copy()

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Feature engineering from datetime
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    # Drop original datetime
    df = df.drop(columns=["datetime"])

    return df

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

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="RandomForest"):
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, test_pred)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print("RandomForest → RMSE:", rmse)

if __name__ == "__main__":
    main()
