import pandas as pd

CLEAN_PATH = "data/processed/cleaned.csv"
OUT_DIR = "data/processed"

def main():
    df = pd.read_csv(CLEAN_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Sort by datetime (critical)
    df = df.sort_values("datetime").reset_index(drop=True)

    total_rows = len(df)
    train_end = int(total_rows * 0.35)
    val_end = int(total_rows * 0.70)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/validate.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Time-based split completed")
    print(f"Total rows     : {total_rows}")
    print(f"Train rows (35%): {len(train_df)}")
    print(f"Validate rows (35%): {len(val_df)}")
    print(f"Test rows (30%) : {len(test_df)}")

    print("\nDate ranges:")
    print("Train   :", train_df['datetime'].min(), "→", train_df['datetime'].max())
    print("Validate:", val_df['datetime'].min(), "→", val_df['datetime'].max())
    print("Test    :", test_df['datetime'].min(), "→", test_df['datetime'].max())

if __name__ == "__main__":
    main()
