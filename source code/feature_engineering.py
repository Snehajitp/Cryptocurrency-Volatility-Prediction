import pandas as pd
import numpy as np

def feature_engineering(df):
    df = df.sort_values(by=["crypto_name", "date"])

    # Returns
    df["daily_return"] = (df["close"] - df["open"]) / df["open"]
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling Volatility (TARGET)
    df["volatility_7d"] = (
        df.groupby("crypto_name")["daily_return"]
        .rolling(7)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Moving Averages
    df["ma_7"] = (
        df.groupby("crypto_name")["close"]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ma_14"] = (
        df.groupby("crypto_name")["close"]
        .rolling(14)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Liquidity
    df["liquidity_ratio"] = df["volume"] / df["marketCap"]

    # Intraday volatility
    df["high_low_spread"] = (df["high"] - df["low"]) / df["low"]

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    processed_df = feature_engineering(df)
    processed_df.to_csv("ddataset.csv", index=False)

    print("Feature engineering completed")
