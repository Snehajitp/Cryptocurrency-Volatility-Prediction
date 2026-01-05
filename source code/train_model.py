import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ddataset.csv")




features = [
    "open", "high", "low", "close",
    "volume", "marketCap",
    "daily_return", "ma_7", "ma_14",
    "liquidity_ratio", "high_low_spread"
]

X = df[features]
y = df["volatility_7d"]
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill missing values
X = X.fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (time-based)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "models/volatility_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model training completed")
