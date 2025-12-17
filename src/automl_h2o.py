import h2o
from h2o.automl import H2OAutoML

TRAIN_PATH = "data/processed/train.csv"

def main():
    # Start H2O
    h2o.init(max_mem_size="2G")

    # Load training data only
    train = h2o.import_file(TRAIN_PATH)

    # Define target and features
    y = "count"
    x = [col for col in train.columns if col != y]

    # AutoML
    aml = H2OAutoML(
        max_runtime_secs=600,   # 10 minutes
        seed=42,
        sort_metric="RMSE"
    )

    aml.train(x=x, y=y, training_frame=train)

    # Print leaderboard
    lb = aml.leaderboard
    print("\n H2O AutoML Leaderboard:")
    print(lb.head(rows=10))

    # Print top 3 models
    print("\n TOP 3 MODELS BY RMSE:")
    print(lb.head(rows=3))

    h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()
