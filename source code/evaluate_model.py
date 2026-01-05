import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("ddataset.csv")

features = [
    "open", "high", "low", "close",
    "volume", "marketCap",
    "daily_return", "ma_7", "ma_14",
    "liquidity_ratio", "high_low_spread"
]

X = df[features]
y = df["volatility_7d"]

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/volatility_model.pkl")

X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

