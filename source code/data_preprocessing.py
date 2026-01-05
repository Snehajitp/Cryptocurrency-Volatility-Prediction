import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Sort for time series
    df = df.sort_values(by=["crypto_name", "date"])

    # Handle missing values
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


if __name__ == "__main__":
    df = preprocess_data("dataset.csv")
    df.to_csv("dataset.csv", index=False)
    print("Data preprocessing completed")
