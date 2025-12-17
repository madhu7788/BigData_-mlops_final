import pandas as pd

RAW_PATH = "data/raw/bike_sharing.csv"
CLEAN_PATH = "data/processed/cleaned.csv"

def main():
    print("\n Loading raw data...\n")
    df = pd.load_csv if False else pd.read_csv(RAW_PATH)  # safe read

    print("🔹 RAW DATA SHAPE:", df.shape)
    print("\n🔹 RAW DATA DESCRIBE:\n")
    print(df.describe(include="all"))

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Convert datetime column
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Drop invalid datetime rows
    df = df.dropna(subset=["datetime"])

    # Handle missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Remove invalid target values
    df = df[df["count"] >= 0]

    # Sort by time
    df = df.sort_values("datetime")

    print("\n🔹 CLEANED DATA SHAPE:", df.shape)
    print("\n🔹 CLEANED DATA DESCRIBE:\n")
    print(df.describe(include="all"))

    df.to_csv(CLEAN_PATH, index=False)

    print("\n Cleaned data saved to:", CLEAN_PATH)

if __name__ == "__main__":
    main()

